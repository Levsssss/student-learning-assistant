from src.agent.config_loader import load_config

config = load_config("agent.yaml")

print("Teacher password:", config.teacher_password)