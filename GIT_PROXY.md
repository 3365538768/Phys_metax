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

## 说明

- 若曾设置过 `127.0.0.1:7891` 等本地代理且已关闭，会导致 `Connection refused`，请改用上述代理或清空环境变量后再试。
- 代理地址以你所在网络为准，如有变更请同步修改本文档。
