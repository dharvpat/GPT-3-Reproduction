# This module handles prompt creation for few-shot learning tasks

def generate_prompts(task, num_shots=3):
    if task == 'text_classification':
        return generate_text_classification_prompts(num_shots)
    elif task == 'question_answering':
        return generate_qa_prompts(num_shots)
    else:
        raise ValueError(f"Unknown task: {task}")

def generate_text_classification_prompts(num_shots):
    # Generate few-shot prompts for text classification tasks
    return [f"Example {i+1}: This is a sample classification prompt." for i in range(num_shots)]

def generate_qa_prompts(num_shots):
    # Generate few-shot prompts for question-answering tasks
    return [f"Example {i+1}: What is the capital of France? Answer: Paris." for i in range(num_shots)]