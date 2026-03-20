#!/usr/bin/env bash
# PhysGaussian：带代理的 Git 拉取 / 推送自动化脚本
#
# 用法：
#   ./scripts/git_sync.sh pull [分支]     # 从 origin 拉取并合并
#   ./scripts/git_sync.sh push [说明] [分支]  # add 全部、commit（有变更时）、push
#   ./scripts/git_sync.sh fetch           # 仅 fetch
#   ./scripts/git_sync.sh status          # 仅查看状态（走代理的 remote 检查可选）
#
# 环境变量（可选）：
#   GIT_HTTP_PROXY   默认 http://proxy.hs.com:3128
#   GIT_BRANCH       默认 main（push/pull 未传分支时使用）
#   SKIP_PROXY=1     不为 git 设置 -c http.proxy（直连时使用）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 默认国内代理；可在执行前 export GIT_HTTP_PROXY=... 覆盖
PROXY="${GIT_HTTP_PROXY:-http://proxy.hs.com:3128}"
BRANCH_DEFAULT="${GIT_BRANCH:-main}"

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

# 使用显式 -c 代理，避免全局里残留的 127.0.0.1:7891 导致 Connection refused
git_proxy() {
  if [[ "${SKIP_PROXY:-0}" == "1" ]]; then
    git "$@"
  else
    git -c "http.proxy=${PROXY}" -c "https.proxy=${PROXY}" "$@"
  fi
}

# 可选：同步环境变量，供 credential helper / 子进程使用
export_proxy_env() {
  if [[ "${SKIP_PROXY:-0}" == "1" ]]; then
    return 0
  fi
  export http_proxy="${PROXY}"
  export https_proxy="${PROXY}"
  export HTTP_PROXY="${PROXY}"
  export HTTPS_PROXY="${PROXY}"
}

cmd_pull() {
  local branch="${1:-${BRANCH_DEFAULT}}"
  export_proxy_env
  echo ">>> git pull origin ${branch}（代理: ${PROXY}）"
  git_proxy pull origin "${branch}"
}

cmd_fetch() {
  export_proxy_env
  echo ">>> git fetch --all --prune（代理: ${PROXY}）"
  git_proxy fetch --all --prune
}

cmd_push() {
  local msg="${1:-}"
  local branch="${2:-${BRANCH_DEFAULT}}"
  export_proxy_env

  echo ">>> git add -A"
  git add -A

  if git diff --cached --quiet; then
    echo ">>> 暂存区无新变更，跳过 commit"
  else
    if [[ -z "${msg}" ]]; then
      msg="chore: update $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo ">>> git commit -m \"${msg}\""
    git commit -m "${msg}"
  fi

  echo ">>> git push -u origin ${branch}（代理: ${PROXY}）"
  git_proxy push -u origin "${branch}"
}

cmd_status() {
  git status
}

main() {
  local sub="${1:-}"
  shift || true
  case "${sub}" in
    pull)  cmd_pull "$@" ;;
    push)  cmd_push "$@" ;;
    fetch) cmd_fetch "$@" ;;
    status) cmd_status "$@" ;;
    ""|-h|--help|help) usage ;;
    *)
      echo "未知子命令: ${sub}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
