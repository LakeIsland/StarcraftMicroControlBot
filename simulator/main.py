from agentEvaluator import AgentEvaluator
from agentTrainer import AgentTrainer

if __name__ == "__main__":
    #eval = AgentEvaluator(fileName = "../q_table_test.txt")
    #eval.evaluate()
    trainer = AgentTrainer(fileName="../q_table_test.txt", maxIterate=500, epsilon_decrease="LINEAR")
    trainer.train()