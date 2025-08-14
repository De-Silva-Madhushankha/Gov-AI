from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict

class AgentState(TypedDict):
    user_input: str
    intent: str
    response: str
    db_result: dict

llm = ChatOllama(model="llama3")

# Node Functions 
def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent based on input"""
    user_input = state["user_input"].lower()
    
    if any(word in user_input for word in ["document", "status", "check", "application"]):
        intent = "check_documents"
    elif any(word in user_input for word in ["complaint", "problem", "issue", "file"]):
        intent = "file_complaint"
    elif any(word in user_input for word in ["appointment", "schedule", "meeting", "book"]):
        intent = "schedule_appointment"
    else:
        intent = "general"
    
    return {**state, "intent": intent}

def fetch_documents(state: AgentState) -> AgentState:
    """Handle document checking requests"""
    # Simulate database lookup
    db_result = {"status": "approved", "documents": ["passport", "license"]}
    response = f"Your documents are {db_result['status']}. Available documents: {', '.join(db_result['documents'])}"
    
    return {**state, "db_result": db_result, "response": response}

def file_complaint(state: AgentState) -> AgentState:
    """Handle complaint filing"""
    # Simulate complaint filing
    complaint_id = "COMP-2025-001"
    db_result = {"complaint_id": complaint_id, "status": "submitted"}
    response = f"Your complaint has been filed successfully. Complaint ID: {complaint_id}"
    
    return {**state, "db_result": db_result, "response": response}

def schedule_appointment(state: AgentState) -> AgentState:
    """Handle appointment scheduling"""
    # Simulate appointment booking
    appointment_id = "APPT-2025-001"
    db_result = {"appointment_id": appointment_id, "date": "2025-08-20", "time": "10:00 AM"}
    response = f"Appointment scheduled successfully. ID: {appointment_id} on {db_result['date']} at {db_result['time']}"
    
    return {**state, "db_result": db_result, "response": response}

def general_response(state: AgentState) -> AgentState:
    """Handle general queries"""
    response = "Hello! I'm here to help you with government services. I can help you check documents, file complaints, or schedule appointments."
    
    return {**state, "response": response}

# Workflow 
workflow = StateGraph(AgentState)

workflow.add_node("classify", classify_intent)
workflow.add_node("check_documents", fetch_documents)
workflow.add_node("file_complaint", file_complaint)
workflow.add_node("schedule_appointment", schedule_appointment)
workflow.add_node("general", general_response)

workflow.set_entry_point("classify")

def route_intent(state: AgentState):
    return state["intent"]

workflow.add_conditional_edges(
    "classify",
    route_intent,
    {
        "check_documents": "check_documents",
        "file_complaint": "file_complaint",
        "schedule_appointment": "schedule_appointment",
        "general": "general"
    }
)

workflow.add_edge("check_documents", END)
workflow.add_edge("file_complaint", END)
workflow.add_edge("schedule_appointment", END)
workflow.add_edge("general", END)

agent = workflow.compile()

# Main execution
if __name__ == "__main__":
    # test_input = "I want to check my document status"
    # result = agent.invoke({"user_input": test_input})
    # print(f"User: {test_input}")
    # print(f"Bot: {result['response']}")

    print("Government Services Chatbot")
    print("Type 'exit' to quit")
    print("-" * 30)
    
    conversation_history = []
    
    user_input = input("Enter: ")
    while user_input != "exit":
        # Store user input in conversation history
        conversation_history.append(f"You: {user_input}")
        
        # Invoke the agent with the correct state structure
        result = agent.invoke({"user_input": user_input})
        
        # Store bot response in conversation history
        bot_response = result['response']
        conversation_history.append(f"Bot: {bot_response}")
        
        # Display the response
        print(f"Bot: {bot_response}")
        
        # Get next input
        user_input = input("Enter: ")

    # Save conversation log
    with open("logging.txt", "w") as file:
        file.write("Your Conversation Log:\n")
        file.write("-" * 30 + "\n")
        
        for message in conversation_history:
            file.write(f"{message}\n")
        
        file.write("-" * 30 + "\n")
        file.write("End of Conversation")

    print("Conversation saved to logging.txt")