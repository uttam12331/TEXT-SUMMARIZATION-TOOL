from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def summarize_text(text, max_length=100):
    """
    Simple text summarization using Ollama with Phi-3
    """
    # Initialize the local LLM
    llm = Ollama(model="phi3")
    
    # Create a simple prompt template
    prompt = PromptTemplate.from_template(
        "Summarize this in {max_length} words or less:\n\n{text}\n\nSummary:"
    )
    
    # Format the prompt with our input
    formatted_prompt = prompt.format(text=text, max_length=max_length)
    
    # Generate the summary
    summary = llm.invoke(formatted_prompt)
    
    return summary.strip()

if __name__ == "__main__":
    print("Simple Text Summarizer")
    print("----------------------\n")
    
    # Example text to summarize
    article = """
    Natural Language Processing (NLP) is a vital subfield of artificial intelligence (AI) and computational linguistics that focuses on enabling machines to understand, interpret, generate, and respond to human language in a meaningful way. From early rule-based translation systems in the 1950s to today’s large transformer-based models like GPT, BERT, and Phi-3, NLP has evolved dramatically through statistical methods and deep learning architectures. It powers countless applications, including virtual assistants, machine translation, sentiment analysis, search engines, chatbots, medical text mining, and automated document summarization. Key tasks in NLP include tokenization, part-of-speech tagging, named entity recognition, dependency parsing, sentiment detection, text summarization, and question answering. These tasks collectively allow machines to process language in ways that mimic human-like understanding. However, NLP still faces major challenges such as language ambiguity, sarcasm detection, contextual understanding, bias in training data, and limitations in supporting low-resource languages. With the rise of large language models, the field is now achieving near-human performance on many benchmarks, yet this also raises concerns around ethical usage, fairness, and environmental impact due to high computational costs. Future directions in NLP include making models more explainable, privacy-focused, and multilingual, while integrating modalities like images and speech for deeper human-computer interaction. As NLP continues to advance, it plays an increasingly central role in transforming how people interact with technology—bridging the gap between human communication and machine logic across sectors like healthcare, law, finance, education, and customer service. Whether it's through summarizing research, interpreting legal documents, assisting in clinical diagnoses, or enhancing accessibility via speech-to-text, NLP is shaping a world where machines better understand the nuances of human language, making information more accessible, actions more efficient, and interactions more intelligent.


    """
    
    print("Original Text:")
    print(article)
    
    print("\nGenerating summary...")
    result = summarize_text(article, max_length=100)
    
    print("\nSummary:")
    print(result)
