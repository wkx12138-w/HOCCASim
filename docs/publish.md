# Publishing this repository on GitHub

The repository is ready to push; nothing has to be generated first.  This page
lists the three ways to publish it, in order of convenience.

---

## Option A — one command with a personal access token (no `gh` CLI needed)

1. Create a token: <https://github.com/settings/tokens>
   * *classic*: scope `repo`
   * *fine-grained*: repository permission **Contents: read and write**
     (plus **Administration: read and write** if you want the script to create
     the repository for you)
2. Run:

```powershell
$env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"
.\tools\publish_to_github.ps1 -User <your-github-name>
```

The script authenticates, creates `hocca-simulator` if it does not exist, pushes
`main`, and then **removes the token from the remote URL again** so it never
ends up in `.git/config`.

Useful switches:

```powershell
.\tools\publish_to_github.ps1 -User <name> -Repo hocca-sim -Private
.\tools\publish_to_github.ps1 -User <name> -Description "my description"
```

3. Clear the token from the shell afterwards:

```powershell
Remove-Item Env:GITHUB_TOKEN
```

## Option B — create the repository in the browser, then push

1. <https://github.com/new> → name `hocca-simulator`, **do not** initialise with
   a README (the repository already has one), public, MIT licence is already in
   the tree.
2. Then:

```bash
git remote add origin https://github.com/<your-name>/hocca-simulator.git
git push -u origin main
```

GitHub will ask for credentials once; use the token from Option A as the
password (or set up an SSH key at <https://github.com/settings/keys> and use
`git@github.com:<your-name>/hocca-simulator.git`).

## Option C — a bundle you can carry anywhere

```bash
git bundle create hocca-simulator.bundle --all
```

The bundle is a single file that contains the whole history; on another machine:

```bash
git clone hocca-simulator.bundle hocca-simulator
cd hocca-simulator
git remote set-url origin https://github.com/<your-name>/hocca-simulator.git
git push -u origin main
```

---

## After the first push

Suggested repository settings:

| Setting | Value |
|---|---|
| About / description | `Reference simulator for HOCCA: communication-constrained cooperative multi-UAV search` |
| Topics | `multi-uav` `marl` `cooperative-search` `option-critic` `value-of-information` `gymnasium` `simulation` |
| Website | documented after the first release |
| Releases | tag `v1.0.0`, attach `hocca-simulator.bundle` and generated `figures/` |
| Actions | enabled — `.github/workflows/tests.yml` runs the 35 unit tests on Linux and Windows for Python 3.9/3.11/3.12 plus a NumPy-only job |

### Recommended release commands

```bash
git tag -a v1.0.0 -m "HOCCA reference simulator v1.0.0"
git push origin v1.0.0
```

### Zenodo DOI (optional)

1. Sign in to <https://zenodo.org> with GitHub.
2. Enable the repository in *Settings → GitHub*.
3. Create a GitHub release; Zenodo then mints a DOI you can put in
   `CITATION.cff`.

---

## Checklist before making it public

- [ ] `README.md` reviewed (it points at the repository URL also used in
      `pyproject.toml` and `CITATION.cff`; update the `wkx12138-w`
      organisation if you use a different name).
- [ ] `LICENSE` holder replaced if institutional wording is required.
- [ ] `CITATION.cff` filled with the real author list and, once available, the
      release DOI.
- [ ] `python -m unittest discover -s tests -v` is green locally.
- [ ] No token, no private path and no draft document is committed — check with
      `git grep -n -i "token\|password"` and `git ls-files`.
- [ ] `docs/environment.md` still matches the current design, especially the
      items marked **RECONSTRUCTED**.
