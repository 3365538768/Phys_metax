#!/bin/bash
# 安装 simple-knn 和 diff-gaussian-rasterization（依赖当前环境的 torch，需先装好 requirements.txt）
# 使用：在 PhysGaussian 目录下执行  bash install_cuda_extensions.sh

set -e
cd "$(dirname "$0")"

echo "检查 torch..."
python -c "import torch; print('torch', torch.__version__)" || { echo "请先安装: pip install -r requirements.txt"; exit 1; }

echo ""
echo "安装 simple-knn（需 CUDA，用当前环境的 torch 编译）..."
pip install --no-build-isolation -e gaussian-splatting/submodules/simple-knn/

echo ""
echo "安装 diff-gaussian-rasterization..."
pip install --no-build-isolation -e gaussian-splatting/submodules/diff-gaussian-rasterization/

echo ""
echo "验证（设置 LD_LIBRARY_PATH 以便加载 libc10.so）..."
TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import simple_knn; import diff_gaussian_rasterization; print('simple_knn 和 diff_gaussian_rasterization 已就绪')"
echo "完成。"
echo ""
echo "若在其它终端直接运行 python 时报 libc10.so 找不到，请先执行："
echo "  export LD_LIBRARY_PATH=\$(python -c \"import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))\"):\$LD_LIBRARY_PATH"
