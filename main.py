"""
企业级智能投研与研报自动化助手 - 主入口
"""
import uuid
from src.agents.graph import run_query


def main():
    """主函数：启动投研助手交互循环"""
    print("欢迎使用企业级智能投研与研报自动化助手")
    print("=" * 50)

    while True:
        user_input = input("\n请输入您的研究需求（输入 'quit' 退出）：\n> ").strip()
        if user_input.lower() in ("quit", "exit", "退出"):
            print("感谢使用，再见！")
            break
        if not user_input:
            continue

        # 每次查询使用独立的 thread_id，支持 Checkpointer 断点续传
        thread_id = str(uuid.uuid4())
        print(f"\n[系统] 任务 ID：{thread_id[:8]}，开始处理...\n")

        report = run_query(question=user_input, thread_id=thread_id)

        print("\n" + "=" * 50)
        print("【研究报告】")
        print(report)
        print("=" * 50)


if __name__ == "__main__":
    main()
