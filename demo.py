"""
Tri-Transformer 演示脚本

展示完整的训练、评估、推理流程
"""
import subprocess
import sys
from pathlib import Path

def print_header(text):
    """打印标题"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"📌 {description}")
    print(f"命令：{cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 成功！\n")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print("❌ 失败！\n")
        if result.stderr:
            print(result.stderr)
        return False

def main():
    """演示主流程"""
    print_header("Tri-Transformer 完整演示")
    
    backend_dir = Path(__file__).parent / "backend"
    
    # 步骤 1: 验证模型
    print_header("步骤 1: 验证模型")
    success = run_command(
        f"cd {backend_dir} && python verify_model.py",
        "运行模型验证脚本"
    )
    
    if not success:
        print("⚠️  模型验证失败，请检查环境配置")
        return
    
    # 步骤 2: 快速训练
    print_header("步骤 2: 快速训练")
    success = run_command(
        f"cd {backend_dir} && python -m app.services.model.quick_start "
        f"--config lightweight --epochs 3 --batch-size 4 --num-samples 200 --max-steps 5",
        "训练轻量级模型（3 epochs）"
    )
    
    if not success:
        print("⚠️  训练失败")
        return
    
    # 步骤 3: 评估模型
    print_header("步骤 3: 评估模型")
    success = run_command(
        f"cd {backend_dir} && python -m app.services.model.evaluate "
        f"--checkpoint ./checkpoints/checkpoint_best.pt --num-samples 50",
        "评估训练好的模型"
    )
    
    if not success:
        print("⚠️  评估失败")
        return
    
    # 步骤 4: 运行测试
    print_header("步骤 4: 运行测试")
    success = run_command(
        f"cd {backend_dir} && python -m pytest tests/test_complete_training.py::TestTriTransformerModel -v",
        "运行模型测试"
    )
    
    if not success:
        print("⚠️  测试失败")
        return
    
    # 完成
    print_header("演示完成！")
    print("✅ 所有步骤成功完成！\n")
    print("📊 生成的文件:")
    print("  - checkpoints/checkpoint_*.pt (模型检查点)")
    print("  - checkpoints/training_history.json (训练历史)")
    print("  - evaluation_results.json (评估结果)")
    print("\n下一步:")
    print("  1. 查看训练历史：cat checkpoints/training_history.json")
    print("  2. 启动交互式推理：python -m app.services.model.inference_cli --checkpoint ./checkpoints/checkpoint_best.pt --mode interactive")
    print("  3. 进行更多训练：python -m app.services.model.quick_start --epochs 10")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 演示中断")
        sys.exit(0)
