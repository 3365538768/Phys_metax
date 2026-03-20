# Git / GitHub 国内代理说明

在终端执行 `git push`、`git fetch` 等访问 GitHub 前，可先设置以下环境变量（本机代理示例）：

```bash
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128
export HTTP_PROXY=http://proxy.hs.com:3128
export HTTPS_PROXY=http://proxy.hs.com:3128
```

然后在该终端内执行 Git 命令，例如：

```bash
cd /path/to/PhysGaussian
git push -u origin main
```

## 仅当前仓库使用代理（可选）

不污染全局环境时，可只给本仓库配置：

```bash
git config http.proxy http://proxy.hs.com:3128
git config https.proxy http://proxy.hs.com:3128
```

取消本仓库代理：

```bash
git config --unset http.proxy
git config --unset https.proxy
```

## 自动化脚本（拉取 / 提交 / 推送）

仓库内提供 **`scripts/git_sync.sh`**，对 `git pull` / `git push` 使用与上文一致的代理（默认 `http://proxy.hs.com:3128`），并用 `git -c http.proxy=...` 覆盖全局里可能错误的 `127.0.0.1:7891`。

```bash
cd /path/to/PhysGaussian
chmod +x scripts/git_sync.sh   # 仅需一次

# 从 origin 拉取当前分支（默认跟踪 main，可传分支名）
./scripts/git_sync.sh pull
./scripts/git_sync.sh pull main

# 暂存全部变更、有变更则 commit、再 push（无说明则用时间戳作为提交说明）
./scripts/git_sync.sh push "docs: 更新说明"
./scripts/git_sync.sh push "fix: xxx" main

# 仅 fetch
./scripts/git_sync.sh fetch

# 查看工作区状态（不走远程，无代理）
./scripts/git_sync.sh status
```

自定义代理或直连：

```bash
export GIT_HTTP_PROXY=http://your-proxy:port
./scripts/git_sync.sh pull

SKIP_PROXY=1 ./scripts/git_sync.sh pull   # 不使用任何 http.proxy
```

## 说明

- 若曾设置过 `127.0.0.1:7891` 等本地代理且已关闭，会导致 `Connection refused`，请改用上述代理或清空环境变量后再试。
- 代理地址以你所在网络为准，如有变更请同步修改本文档或设置 `GIT_HTTP_PROXY`。
- `push` 会执行 `git add -A`；大目录、子模块若有未提交改动请自行检查。`auto_output/`、`my_model/checkpoints/` 等已在 `.gitignore` 中排除。
