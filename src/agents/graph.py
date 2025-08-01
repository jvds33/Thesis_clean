"""
ACOS Pipeline implementation using LangGraph.

This module implements the Aspect-based sentiment analysis pipeline using LangGraph.
The pipeline consists of the following agents:
1. extract_aspects - Extracts aspect terms from the text
2. extract_opinions - Extracts opinion terms and their associated aspects
3. categorize_aspects - Classifies aspects into predefined categories
4. decide_sentiment - Determines sentiment polarity for aspect-opinion pairs
5. link_and_check - Links aspects, opinions, categories, and sentiments into final quadruples

The shared state that flows through the pipeline is represented by a GraphState TypedDict.
"""

import os
import sys
from typing import Dict, List, TypedDict, Optional, Literal, Tuple, Union, Any
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# Constants and categories
from categories import VALID_RESTAURANT_CATEGORIES, VALID_LAPTOP_CATEGORIES

# TypedDict for the graph state
class GraphState(TypedDict, total=False):
    text: str                   # The input sentence
    lang: str                   # Language (e.g., "en")
    aspects: List[str]          # List of identified aspect terms
    opinions: List[Dict]        # List of opinions with their sentiments
    categories: Dict[str, str]  # A map from an aspect to its category
    sentiments: List[Dict]      # List of sentiments linked to aspects/opinions
    quadruples: List[Dict]      # The final structured output
    needs_fix: bool             # A flag for looping, not used in the default flow
    issues: List[str]           # A list of problems found by the linker/checker
    domain: str                 # Domain: "restaurant" or "laptop"

# Response models for the different agents
class AspectExtractionResponse(BaseModel):
    aspects: List[str] = Field(description="List of extracted aspect terms")

class OpinionExtraction(BaseModel):
    aspect: Optional[str] = Field(description="The aspect term, or None for implicit aspects")
    opinion: str = Field(description="The opinion phrase/word about this aspect")

class OpinionExtractionResponse(BaseModel):
    opinions: List[OpinionExtraction] = Field(description="List of extracted opinions with their associated aspects")

class CategoryClassificationResponse(BaseModel):
    categories: Dict[str, str] = Field(description="Dictionary mapping aspect terms to category labels")

class SentimentItem(BaseModel):
    aspect: Optional[str] = Field(description="The aspect term, or None for implicit aspects")
    opinion: str = Field(description="The opinion phrase/word about this aspect")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment polarity")

class SentimentClassificationResponse(BaseModel):
    sentiments: List[SentimentItem] = Field(description="List of aspect-opinion pairs with their sentiment classification")

class Quadruple(BaseModel):
    aspect_term: Optional[str] = Field(description="The aspect term, or null for implicit aspects")
    category: str = Field(description="The category of the aspect")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment polarity")
    opinion: str = Field(description="The opinion phrase/word about this aspect")

class LinkAndCheckResponse(BaseModel):
    quadruples: List[Quadruple] = Field(description="List of quadruples with linked aspects, categories, sentiments, and opinions")
    issues: List[str] = Field(description="List of issues found during linking and checking")

# Define the agent functions
def extract_aspects(state: GraphState) -> GraphState:
    """
    Extract aspect terms from the review text.
    
    Args:
        state: The current graph state
        
    Returns:
        Updated state with 'aspects' key added
    """
    if 'text' not in state:
        raise ValueError("Input state must contain 'text' key")
        
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
        temperature=0
    )
    
    # Set up output parser
    parser = PydanticOutputParser(pydantic_object=AspectExtractionResponse)
    
    # Few-shot examples for restaurant domain
    rest_examples = """
EXAMPLE 1
Text: "The pasta was delicious but service was slow."
Aspects: ["pasta", "service"]

EXAMPLE 2
Text: "I love this restaurant!"
Aspects: ["restaurant"]

EXAMPLE 3
Text: "The staff was friendly and the atmosphere was cozy."
Aspects: ["staff", "atmosphere"]

EXAMPLE 4
Text: "I didn't like it."
Aspects: ["null"]
"""

    # Few-shot examples for laptop domain
    laptop_examples = """
EXAMPLE 1
Text: "The battery life is excellent but the keyboard feels cheap."
Aspects: ["battery life", "keyboard"]

EXAMPLE 2
Text: "This laptop is fast and reliable."
Aspects: ["laptop"]

EXAMPLE 3
Text: "The screen resolution is amazing but the touchpad is not responsive."
Aspects: ["screen resolution", "touchpad"]

EXAMPLE 4
Text: "I hate it."
Aspects: ["null"]
"""

    examples = rest_examples if state.get('domain', 'restaurant') == 'restaurant' else laptop_examples
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an aspect extraction specialist. Extract explicit aspect terms from the review text.\n\n"
         "Guidelines:\n"
         "1. Extract nouns or noun phrases that are being evaluated or commented on.\n"
         "2. Extract aspects exactly as they appear in the text - preserve spelling and case.\n"
         "3. If there are no explicit aspects, return ['null'].\n"
         "4. Focus only on aspects that have associated opinions.\n\n"
         "Here are examples of proper aspect extraction:\n"
         "{examples}\n\n"
         "{format_instructions}"
        ),
        ("human", "Review text: {text}")
    ])
    
    prompt = prompt.partial(
        examples=examples,
        format_instructions=parser.get_format_instructions()
    )
    
    # Run the chain
    chain = prompt | llm | parser
    result = chain.invoke({"text": state['text']})
    
    # Update state with extracted aspects
    state['aspects'] = result.aspects
    
    return state

def extract_opinions(state: GraphState) -> GraphState:
    """
    Extract opinions and their associated aspects from the review text.
    
    Args:
        state: The current graph state with 'text' and 'aspects' keys
        
    Returns:
        Updated state with 'opinions' key added
    """
    if 'text' not in state or 'aspects' not in state:
        raise ValueError("Input state must contain 'text' and 'aspects' keys")
        
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
        temperature=0
    )
    
    # Set up output parser
    parser = PydanticOutputParser(pydantic_object=OpinionExtractionResponse)
    
    # Few-shot examples for restaurant domain
    rest_examples = """
EXAMPLE 1
Text: "The pasta was delicious but service was slow."
Aspects: ["pasta", "service"]
Opinions: [{"aspect":"pasta","opinion":"delicious"},{"aspect":"service","opinion":"slow"}]

EXAMPLE 2
Text: "I love this restaurant!"
Aspects: ["restaurant"]
Opinions: [{"aspect":"restaurant","opinion":"love"}]

EXAMPLE 3
Text: "The staff was friendly and the atmosphere was cozy."
Aspects: ["staff", "atmosphere"]
Opinions: [{"aspect":"staff","opinion":"friendly"},{"aspect":"atmosphere","opinion":"cozy"}]

EXAMPLE 4
Text: "I didn't like it."
Aspects: ["null"]
Opinions: [{"aspect":null,"opinion":"didn't like"}]
"""

    # Few-shot examples for laptop domain
    laptop_examples = """
EXAMPLE 1
Text: "The battery life is excellent but the keyboard feels cheap."
Aspects: ["battery life", "keyboard"]
Opinions: [{"aspect":"battery life","opinion":"excellent"},{"aspect":"keyboard","opinion":"cheap"}]

EXAMPLE 2
Text: "This laptop is fast and reliable."
Aspects: ["laptop"]
Opinions: [{"aspect":"laptop","opinion":"fast"},{"aspect":"laptop","opinion":"reliable"}]

EXAMPLE 3
Text: "The screen resolution is amazing but the touchpad is not responsive."
Aspects: ["screen resolution", "touchpad"]
Opinions: [{"aspect":"screen resolution","opinion":"amazing"},{"aspect":"touchpad","opinion":"not responsive"}]

EXAMPLE 4
Text: "I hate it."
Aspects: ["null"]
Opinions: [{"aspect":null,"opinion":"hate"}]
"""

    examples = rest_examples if state.get('domain', 'restaurant') == 'restaurant' else laptop_examples
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an opinion extraction specialist for reviews. "
         "Extract opinion phrases associated with each aspect in the text. "
         "An opinion phrase expresses sentiment or evaluation about the aspect. "
         "Opinions are typically adjectives, adverbs, or short phrases that describe qualities.\n\n"
         "Guidelines:\n"
         "1. Extract opinions exactly as they appear in the text - preserve spelling and case.\n"
         "2. For each aspect, extract all opinion phrases that directly describe it.\n"
         "3. If an aspect appears multiple times with different opinions, include all opinions.\n"
         "4. Focus on adjectives, verbs, and descriptive phrases that convey sentiment.\n"
         "5. Include negations (e.g., 'not good', 'didn't like') as part of the opinion phrase.\n\n"
         "Here are examples of proper opinion extraction:\n"
         "{examples}\n\n"
         "{format_instructions}"
        ),
        ("human", "Review text: {text}\nAspects: {aspects}")
    ])
    
    prompt = prompt.partial(
        examples=examples,
        format_instructions=parser.get_format_instructions()
    )
    
    # If there are no aspects or only null aspect, return empty opinions
    if not state['aspects'] or (len(state['aspects']) == 1 and state['aspects'][0] == "null"):
        state['opinions'] = []
        return state
    
    # Run the chain
    chain = prompt | llm | parser
    result = chain.invoke({
        "text": state['text'],
        "aspects": state['aspects']
    })
    
    # Update state with extracted opinions
    state['opinions'] = [{"aspect": opinion.aspect, "opinion": opinion.opinion} for opinion in result.opinions]
    
    return state

def categorize_aspects(state: GraphState) -> GraphState:
    """
    Classify aspects into predefined categories.
    
    Args:
        state: The current graph state with 'text' and 'aspects' keys
        
    Returns:
        Updated state with 'categories' key added
    """
    if 'text' not in state or 'aspects' not in state:
        raise ValueError("Input state must contain 'text' and 'aspects' keys")
        
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
        temperature=0
    )
    
    # Set up output parser
    parser = PydanticOutputParser(pydantic_object=CategoryClassificationResponse)
    
    # Define category lists based on domain
    domain = state.get('domain', 'restaurant')
    valid_categories = VALID_RESTAURANT_CATEGORIES if domain == 'restaurant' else VALID_LAPTOP_CATEGORIES
    
    # Restaurant category examples
    rest_examples = """
EXAMPLE 1
Text: "The pasta was delicious but service was slow."
Aspects: ["pasta", "service"]
Categories: {"pasta": "food quality", "service": "service general"}

EXAMPLE 2
Text: "The restaurant has a nice ambiance but is quite expensive."
Aspects: ["ambiance", "restaurant"]
Categories: {"ambiance": "ambience general", "restaurant": "restaurant prices"}

EXAMPLE 3
Text: "I love their wine selection."
Aspects: ["wine selection"]
Categories: {"wine selection": "drinks style_options"}

EXAMPLE 4
Text: "I had a terrible experience."
Aspects: ["null"]
Categories: {"null": "restaurant general"}
"""
    
    # Laptop category examples
    laptop_examples = """
EXAMPLE 1
Text: "The battery life is excellent but the keyboard feels cheap."
Aspects: ["battery life", "keyboard"]
Categories: {"battery life": "battery general", "keyboard": "keyboard quality"}

EXAMPLE 2
Text: "The laptop is fast but overheats quickly."
Aspects: ["laptop", "laptop"]
Categories: {"laptop": "laptop operation_performance"}

EXAMPLE 3
Text: "The screen resolution is amazing but Windows keeps crashing."
Aspects: ["screen resolution", "Windows"]
Categories: {"screen resolution": "display quality", "Windows": "os operation_performance"}

EXAMPLE 4
Text: "I hate it."
Aspects: ["null"]
Categories: {"null": "laptop general"}
"""
    
    examples = rest_examples if domain == 'restaurant' else laptop_examples
    category_list = "\n".join([f"• {category}" for category in valid_categories])
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a category classification specialist for {domain} reviews. "
         "Classify each aspect term into one of the predefined categories.\n\n"
         "Valid categories:\n{category_list}\n\n"
         "Guidelines:\n"
         "1. Assign exactly one category to each aspect.\n"
         "2. Choose the most specific category that applies.\n"
         "3. If an aspect is implicit (null), use the general category for the domain.\n"
         "4. Consider the context of the review when classifying.\n\n"
         "Here are examples of proper category classification:\n"
         "{examples}\n\n"
         "{format_instructions}"
        ),
        ("human", "Review text: {text}\nAspects: {aspects}")
    ])
    
    prompt = prompt.partial(
        domain=domain,
        category_list=category_list,
        examples=examples,
        format_instructions=parser.get_format_instructions()
    )
    
    # Run the chain
    chain = prompt | llm | parser
    result = chain.invoke({
        "text": state['text'],
        "aspects": state['aspects']
    })
    
    # Validate that all categories are valid
    for aspect, category in result.categories.items():
        if category not in valid_categories:
            raise ValueError(f"Invalid category '{category}' for aspect '{aspect}'")
    
    # Update state with category classifications
    state['categories'] = result.categories
    
    return state

def decide_sentiment(state: GraphState) -> GraphState:
    """
    Determine sentiment polarity for aspect-opinion pairs.
    
    Args:
        state: The current graph state with 'text', 'aspects', and 'opinions' keys
        
    Returns:
        Updated state with 'sentiments' key added
    """
    if 'text' not in state or 'opinions' not in state:
        raise ValueError("Input state must contain 'text' and 'opinions' keys")
        
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
        temperature=0
    )
    
    # Set up output parser
    parser = PydanticOutputParser(pydantic_object=SentimentClassificationResponse)
    
    # Few-shot examples
    examples = """
EXAMPLE 1 (Implicit Positive)
Text: "I can't wait to come back!"
Pairs:
- aspect: null, opinion: "can't wait"
Output:
{"sentiments":[{"aspect":null,"opinion":"can't wait","sentiment":"positive"}]}

EXAMPLE 2 (Negative with Modifier)
Text: "The food was rather bland today."
Pairs:
- aspect: "food", opinion: "rather bland"
Output:
{"sentiments":[{"aspect":"food","opinion":"rather bland","sentiment":"negative"}]}

EXAMPLE 3 (Neutral)
Text: "Service felt adequate overall."
Pairs:
- aspect: "service", opinion: "adequate"
Output:
{"sentiments":[{"aspect":"service","opinion":"adequate","sentiment":"neutral"}]}

EXAMPLE 4 (Implicit Negative—use full text)
Text: "No complaints, but it wasn't great either."
Pairs:
- aspect: null, opinion: "no complaints"
Output:
{"sentiments":[{"aspect":null,"opinion":"no complaints","sentiment":"positive"}]}
"""
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a sentiment analysis expert. Your task is to classify the sentiment of opinions about aspects in a review.\n\n"
         "Guidelines:\n"
         "1. Classify each opinion as exactly one of: 'positive', 'negative', or 'neutral'.\n"
         "2. Base your classification primarily on the opinion phrase, not the full text.\n"
         "3. If the opinion is 'null', refer to the full text to determine sentiment toward that aspect.\n"
         "4. Classify as 'neutral' when the opinion expresses factual information without evaluation.\n"
         "5. Focus on the semantic meaning in context, not just individual words.\n\n"
         "Here are examples of proper sentiment classification:\n"
         "{examples}\n\n"
         "{format_instructions}"
        ),
        ("human", 
         "Full review text: {text}\n\n"
         "Opinions to classify: {opinions}\n\n"
         "Language: {lang}\n\n"
         "Classify the sentiment of each aspect-opinion pair:")
    ])
    
    prompt = prompt.partial(
        examples=examples,
        format_instructions=parser.get_format_instructions()
    )
    
    # Set default language if not provided
    lang = state.get('lang', 'en')
    
    # If there are no opinions, return empty sentiments
    if not state['opinions']:
        state['sentiments'] = []
        return state
    
    # Run the chain
    chain = prompt | llm | parser
    result = chain.invoke({
        "text": state['text'],
        "opinions": state['opinions'],
        "lang": lang
    })
    
    # Update state with sentiment classifications
    state['sentiments'] = [
        {"aspect": item.aspect, "opinion": item.opinion, "sentiment": item.sentiment} 
        for item in result.sentiments
    ]
    
    return state

def link_and_check(state: GraphState) -> GraphState:
    """
    Link aspects, opinions, categories, and sentiments into final quadruples.
    
    Args:
        state: The current graph state with all required keys
        
    Returns:
        Updated state with 'quadruples' and 'issues' keys added
    """
    if not all(k in state for k in ['text', 'aspects', 'opinions', 'categories', 'sentiments']):
        raise ValueError("State must contain 'text', 'aspects', 'opinions', 'categories', and 'sentiments' keys")
        
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
        temperature=0
    )
    
    # Set up output parser
    parser = PydanticOutputParser(pydantic_object=LinkAndCheckResponse)
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a data integration specialist. Your task is to link aspects, categories, opinions, and sentiments "
         "into complete quadruples and check for any inconsistencies or issues.\n\n"
         "Guidelines:\n"
         "1. Create a quadruple for each aspect-opinion pair, with the corresponding category and sentiment.\n"
         "2. For implicit aspects (null), use the general category for the domain.\n"
         "3. Check for missing elements or inconsistencies.\n"
         "4. Report any issues found.\n\n"
         "{format_instructions}"
        ),
        ("human", 
         "Review text: {text}\n\n"
         "Aspects: {aspects}\n\n"
         "Categories: {categories}\n\n"
         "Opinions: {opinions}\n\n"
         "Sentiments: {sentiments}\n\n"
         "Link all these elements into quadruples and check for issues:")
    ])
    
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    # Create quadruples directly without LLM for simple cases
    quadruples = []
    issues = []
    
    # Helper function to convert None to "null" for consistent representation
    def normalize_aspect(aspect):
        return aspect if aspect is not None else "null"
    
    # For each sentiment item, create a quadruple
    for sentiment_item in state['sentiments']:
        aspect = sentiment_item['aspect']
        opinion = sentiment_item['opinion']
        sentiment = sentiment_item['sentiment']
        
        # Find the category for this aspect
        category = state['categories'].get(normalize_aspect(aspect), "unknown")
        
        if category == "unknown":
            issues.append(f"Could not find category for aspect: {normalize_aspect(aspect)}")
        
        # Add quadruple
        quadruples.append({
            "aspect_term": aspect,
            "category": category,
            "opinion": opinion,
            "sentiment": sentiment
        })
    
    # If we have opinions but no quadruples, there might be an issue with linking
    if state['opinions'] and not quadruples:
        issues.append("Failed to create any quadruples despite having opinions.")
    
    # Update state with quadruples and issues
    state['quadruples'] = quadruples
    state['issues'] = issues
    
    return state

# Create the LangGraph pipeline
def create_pipeline():
    """
    Create and return the ACOS pipeline as a LangGraph.
    
    Returns:
        A LangGraph pipeline for the ACOS task
    """
    # Create the graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("extract_aspects", extract_aspects)
    graph.add_node("extract_opinions", extract_opinions)
    graph.add_node("categorize_aspects", categorize_aspects)
    graph.add_node("decide_sentiment", decide_sentiment)
    graph.add_node("link_and_check", link_and_check)
    
    # Add edges to connect the nodes in sequence
    graph.add_edge("extract_aspects", "extract_opinions")
    graph.add_edge("extract_opinions", "categorize_aspects")
    graph.add_edge("categorize_aspects", "decide_sentiment")
    graph.add_edge("decide_sentiment", "link_and_check")
    
    # Set the entry and end points
    graph.set_entry_point("extract_aspects")
    graph.set_finish_point("link_and_check")
    
    # Compile the graph
    return graph.compile()

def run_pipeline(text: str, domain: str = "restaurant", lang: str = "en") -> Dict[str, Any]:
    """
    Run the ACOS pipeline on the given text.
    
    Args:
        text: The text to analyze
        domain: The domain of the text ("restaurant" or "laptop")
        lang: The language of the text
        
    Returns:
        The final state containing all analysis results
    """
    # Create initial state
    state: GraphState = {
        "text": text,
        "domain": domain,
        "lang": lang
    }
    
    # Create and run the pipeline
    pipeline = create_pipeline()
    final_state = pipeline.invoke(state)
    
    return final_state

def format_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the state into a structured output format.
    
    Args:
        state: The state dictionary after pipeline completion
        
    Returns:
        Structured results dictionary
    """
    results = {
        "text": state["text"],
        "acos_results": []
    }
    
    # Add formatted quadruples to results
    for quadruple in state.get("quadruples", []):
        results["acos_results"].append({
            "aspect": quadruple["aspect_term"],
            "category": quadruple["category"],
            "opinion": quadruple["opinion"],
            "sentiment": quadruple["sentiment"]
        })
    
    return results 