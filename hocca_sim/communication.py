"""Evidence ledger, value-of-information scheduler and provenance fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beliefs import kl_bernoulli_after_item, sigmoid
from .config import CommunicationConfig

__all__ = [
    "EvidenceItem", "Packet", "DeliveryReport", "CommunicationSystem", "score_items",
]


@dataclass
class EvidenceItem:
    """One directed innovation with a unique provenance identifier."""

    provenance: int
    sender: int
    receiver: int
    x: int
    y: int
    step: int
    delta: float
    score: float = 0.0

    @property
    def cell(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Packet:
    """Up to ``m`` items from one sender to one receiver."""

    sender: int
    receiver: int
    send_step: int
    items: List[EvidenceItem] = field(default_factory=list)
    delivered_step: Optional[int] = None
    lost: bool = False

    @property
    def size(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


@dataclass
class DeliveryReport:
    """Result of one communication round."""

    fused: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    duplicates: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    senders: Dict[int, List[int]] = field(default_factory=dict)
    sender_counts: Dict[int, Dict[int, int]] = field(default_factory=dict)
    packets_sent: int = 0
    packets_delivered: int = 0
    packets_lost: int = 0
    items_attempted: int = 0
    items_delivered: int = 0
    items_fused: int = 0
    application_bytes: int = 0

    def empty(self) -> bool:
        return self.items_attempted == 0 and not self.fused


def score_items(
    receiver_prior_log_odds: np.ndarray,
    cells: np.ndarray,
    deltas: np.ndarray,
    receivers: Sequence[int],
    ack_ell: np.ndarray,
) -> List[tuple]:
    del ack_ell
    out: List[tuple] = []
    if len(cells) == 0 or len(receivers) == 0:
        return out
    for r in receivers:
        q = sigmoid(receiver_prior_log_odds[r][cells[:, 1], cells[:, 0]])
        scores = kl_bernoulli_after_item(q, deltas)
        for idx, s in enumerate(scores):
            out.append((float(s), int(idx), int(r)))
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


class CommunicationSystem:
    """Ledger, scheduler, impaired channel and once-only fusion."""

    def __init__(
        self,
        cfg: CommunicationConfig,
        grid_size: int,
        agent_count: int,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.grid_size = int(grid_size)
        self.n_agents = int(agent_count)
        self.rng = rng
        self._next_provenance = 0
        self.ledger: List[List[tuple]] = [[] for _ in range(self.n_agents)]
        self.applied_ids: List[set] = [set() for _ in range(self.n_agents)]
        self.ack_ell = np.zeros(
            (self.n_agents, self.n_agents, self.grid_size, self.grid_size),
            dtype=np.float64,
        )
        self.in_flight: List[Packet] = []
        self.total_items_attempted = 0
        self.total_items_delivered = 0
        self.total_packets_attempted = 0
        self.total_packets_lost = 0
        self.total_bytes = 0

    def reset(self, prior_log_odds: float) -> None:
        self._next_provenance = 0
        self.ledger = [[] for _ in range(self.n_agents)]
        self.applied_ids = [set() for _ in range(self.n_agents)]
        self.ack_ell.fill(float(prior_log_odds))
        self.in_flight.clear()
        self.total_items_attempted = 0
        self.total_items_delivered = 0
        self.total_packets_attempted = 0
        self.total_packets_lost = 0
        self.total_bytes = 0

    def new_provenance(self) -> int:
        pid = self._next_provenance
        self._next_provenance += 1
        return pid

    def record_observations(
        self, agent: int, step: int, cells: np.ndarray, deltas: np.ndarray
    ) -> None:
        if len(cells) == 0:
            return
        queue = self.ledger[agent]
        for (x, y), d in zip(cells, deltas):
            queue.append((int(x), int(y), int(step), float(d), set()))

    def expire(self, step: int) -> None:
        window = self.cfg.replay_window
        eligible = max(self.n_agents - 1, 1)
        for agent in range(self.n_agents):
            queue = self.ledger[agent]
            if not queue:
                continue
            self.ledger[agent] = [
                item for item in queue
                if step - item[2] <= window and len(item[4]) < eligible
            ]

    def schedule(
        self,
        step: int,
        sender: int,
        own_log_odds: np.ndarray,
        active: np.ndarray,
        bandwidth: Optional[int] = None,
    ) -> List[Packet]:
        """Build one sender's packets for one step (modular top-Q rule)."""

        del own_log_odds
        budget = self.cfg.budget_at(step) if bandwidth is None else int(bandwidth)
        if budget <= 0 or not self.cfg.link_available(step):
            return []
        queue = self.ledger[sender]
        if not queue:
            return []

        receivers = [
            r for r in range(self.n_agents)
            if (r != sender or self.cfg.allow_self_loop) and active[r]
        ]
        if not receivers:
            return []

        cells = np.array([(q[0], q[1]) for q in queue], dtype=np.int64)
        deltas = np.array([q[3] for q in queue], dtype=np.float64)
        cost = float(self.cfg.item_cost) * float(self.cfg.bit_cost_lambda)

        per_receiver: List[Tuple[float, int, List[EvidenceItem]]] = []
        for receiver in receivers:
            prior = sigmoid(self.ack_ell[sender, receiver][cells[:, 1], cells[:, 0]])
            scores = kl_bernoulli_after_item(prior, deltas)
            net = scores - cost
            eligible = np.array(
                [receiver not in queue[i][4] for i in range(len(queue))], dtype=bool
            )
            if self.cfg.positive_net_score_only:
                eligible &= net > 0.0
            keep = np.flatnonzero(eligible)
            if len(keep) == 0:
                continue
            chosen = keep[np.argsort(-net[keep], kind="stable")][: self.cfg.items_per_packet]
            items = [
                EvidenceItem(
                    provenance=self.new_provenance(),
                    sender=sender,
                    receiver=receiver,
                    x=int(cells[i, 0]),
                    y=int(cells[i, 1]),
                    step=int(queue[i][2]),
                    delta=float(queue[i][3]),
                    score=float(scores[i]),
                )
                for i in chosen
            ]
            for i in chosen:
                queue[int(i)][4].add(receiver)
            per_receiver.append((float(net[chosen].sum()), receiver, items))

        if not per_receiver:
            return []
        per_receiver.sort(key=lambda t: (-t[0], t[1]))
        return [
            Packet(sender=sender, receiver=receiver, send_step=step, items=items)
            for _, receiver, items in per_receiver[:budget]
        ]

    def transmit(self, step: int, packets: Sequence[Packet]) -> DeliveryReport:
        report = DeliveryReport()
        cfg = self.cfg
        for packet in packets:
            self.total_packets_attempted += 1
            self.total_items_attempted += packet.size
            report.packets_sent += 1
            report.items_attempted += packet.size
            size = cfg.header_bytes + packet.size * cfg.item_bytes
            report.application_bytes += size
            self.total_bytes += size

            if not cfg.link_available(step):
                packet.lost = True
            elif cfg.packet_loss > 0.0 and self.rng.random() < cfg.packet_loss:
                packet.lost = True

            if packet.lost:
                self.total_packets_lost += 1
                report.packets_lost += 1
                continue

            packet.delivered_step = step + 1 + int(cfg.delay_steps)
            self.in_flight.append(packet)

        self._release(step, report)
        return report

    def _release(self, step: int, report: DeliveryReport) -> None:
        still: List[Packet] = []
        for packet in self.in_flight:
            if packet.delivered_step is None or packet.delivered_step > step:
                still.append(packet)
                continue
            self._deliver(packet, report)
        self.in_flight = still

    def _deliver(self, packet: Packet, report: DeliveryReport) -> None:
        receiver = packet.receiver
        report.packets_delivered += 1
        self.total_items_delivered += packet.size
        report.items_delivered += packet.size
        report.senders.setdefault(receiver, []).append(packet.sender)
        counts = report.sender_counts.setdefault(receiver, {})
        counts[packet.sender] = counts.get(packet.sender, 0) + packet.size

        fresh_cells: List[Tuple[int, int]] = []
        fresh_deltas: List[float] = []
        dup_cells: List[Tuple[int, int]] = []
        dup_deltas: List[float] = []

        applied = self.applied_ids[receiver]
        for item in packet.items:
            if item.provenance in applied:
                dup_cells.append(item.cell)
                dup_deltas.append(item.delta)
                continue
            applied.add(item.provenance)
            fresh_cells.append(item.cell)
            fresh_deltas.append(item.delta)

        column = self.ack_ell[packet.sender, receiver]
        for item in packet.items:
            column[item.y, item.x] += item.delta
        np.clip(column, -1e6, 1e6, out=column)

        if fresh_cells:
            report.fused[receiver] = (
                np.asarray(fresh_cells, dtype=np.int64),
                np.asarray(fresh_deltas, dtype=np.float64),
            )
            report.items_fused += len(fresh_cells)
            ack = self.cfg.acknowledgement_bytes(len(fresh_cells))
            report.application_bytes += ack
            self.total_bytes += ack
        if dup_cells:
            report.duplicates[receiver] = (
                np.asarray(dup_cells, dtype=np.int64),
                np.asarray(dup_deltas, dtype=np.float64),
            )

    def summary(self) -> Dict[str, float]:
        return {
            "packets_attempted": float(self.total_packets_attempted),
            "packets_lost": float(self.total_packets_lost),
            "items_attempted": float(self.total_items_attempted),
            "items_delivered": float(self.total_items_delivered),
            "application_bytes": float(self.total_bytes),
            "unique_provenance_ids": float(self._next_provenance),
        }
