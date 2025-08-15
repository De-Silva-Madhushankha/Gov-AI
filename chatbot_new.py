from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict, Optional
import json
from datetime import datetime
from supabase import create_client, Client
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")

class AgentState(TypedDict):
    user_input: str
    intent: str
    response: str
    db_result: dict
    service_name: Optional[str]
    user_context: Optional[dict]

try:
    llm = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini model initialized successfully")
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    llm = None

def get_supabase_client() -> Client:
    """Get Supabase client connection"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return None

def test_connections():
    """Test both Gemini and Supabase connections"""
    print("Testing connections...")
    
    # Test Gemini
    if not GEMINI_API_KEY:
        print("Missing GEMINI_API_KEY in .env file")
        return False
    
    if llm is None:
        print("Gemini model initialization failed")
        return False
    
    try:
        # Test Gemini with a simple prompt
        response = llm.generate_content("Hello")
        print("Gemini connection successful")
    except Exception as e:
        print(f"Gemini connection failed: {e}")
        return False
    
    # Test Supabase
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Missing Supabase credentials in .env file")
        return False
    
    client = get_supabase_client()
    if client is None:
        print("Supabase connection failed.")
        return False
    
    try:
        response = client.table("service").select("*").limit(1).execute()
        print("Supabase connection successful.")
        print(f"Found {len(response.data)} records in Service table")
        return True
    except Exception as e:
        print(f"Supabase query failed: {e}")
        return False

def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent using Gemini AI"""
    user_input = state["user_input"]
    
    # If Gemini is not available, fall back to rule-based classification
    # if llm is None:
    #     return classify_intent_fallback(state)
    
    try:
        prompt = f"""
        Classify the following user query about Sri Lankan government services into one of these categories:

        Categories:
        - service_requirements: User asking about required documents or requirements for a specific service
        - check_documents: User wants to check status of their application or documents
        - schedule_appointment: User wants to book an appointment or meeting
        - file_complaint: User wants to file a complaint or report an issue
        - general_info: User asking for general information about government services
        - general: Any other general queries

        User query: "{user_input}"

        Respond with only the category name, nothing else.
        """
        
        response = llm.generate_content(prompt)
        intent = response.text.strip().lower()
        
        # Validate the response is one of our expected intents
        valid_intents = ["service_requirements", "check_documents", "schedule_appointment", 
                        "file_complaint", "general_info", "general"]
        
        if intent not in valid_intents:
            intent = "general"
            
    except Exception as e:
        print(f"Error with Gemini classification: {e}")
        # Fall back to rule-based classification
        # return classify_intent_fallback(state)
    
    return {**state, "intent": intent}

# def classify_intent_fallback(state: AgentState) -> AgentState:
#     """Fallback rule-based intent classification"""
#     user_input = state["user_input"].lower()
    
#     # Service document inquiry patterns
#     service_keywords = ["document", "documents", "need", "required", "requirements", "paperwork"]
#     service_patterns = ["what documents", "what do i need", "requirements for", "documents for"]
    
#     # Check if asking about service requirements
#     if any(pattern in user_input for pattern in service_patterns) or \
#        (any(keyword in user_input for keyword in service_keywords) and 
#         any(service in user_input for service in ["passport", "license", "certificate", "registration", "permit"])):
#         intent = "service_requirements"
    
#     # Document status checking
#     elif any(word in user_input for word in ["status", "check", "application", "my documents"]):
#         intent = "check_documents"
    
#     # Appointment related
#     elif any(word in user_input for word in ["appointment", "schedule", "meeting", "book", "slot"]):
#         intent = "schedule_appointment"
    
#     # Complaint filing
#     elif any(word in user_input for word in ["complaint", "problem", "issue", "file complaint"]):
#         intent = "file_complaint"
    
#     # General government services info
#     elif any(word in user_input for word in ["help", "services", "what can", "how do"]):
#         intent = "general_info"
    
#     else:
#         intent = "general"
    
#     return {**state, "intent": intent}

# classify_intent = classify_intent_with_gemini

def extract_service_name(user_input: str) -> Optional[str]:
    """Extract service name from user input"""
    user_input = user_input.lower()
    
    # Common service keywords that might be in your database
    service_keywords = [
        "passport", "driving license", "license", "birth certificate", 
        "marriage certificate", "business registration", "tax clearance",
        "police clearance", "grama niladhari", "identity card", "nic",
        "death certificate", "divorce certificate", "land registration"
    ]
    
    for keyword in service_keywords:
        if keyword in user_input:
            return keyword
    
    return None

# Update the get_service_requirements function
def get_service_requirements(state: AgentState) -> AgentState:
    """Get required documents for a specific service"""
    user_input = state["user_input"]
    service_name = extract_service_name(user_input)
    
    if not service_name:
        response = """I'd be happy to help you find the required documents for a service. 
        
Could you please specify which service you need information about? For example:
• Passport application
• Driving license
• Birth certificate
• Marriage certificate
• Business registration
• Tax clearance certificate

Just say something like "What documents do I need for a passport?" """
        
        return {**state, "response": response, "service_name": None}
    
    try:
        client = get_supabase_client()
        if client is None:
            raise Exception("Could not connect to database")
        
        # Get service information
        service_response = client.table("service").select(
            "service_id, title, description, department_id, department(title, email, phone_no)"
        ).ilike("title", f"%{service_name}%").execute()
        
        if not service_response.data:
            response = f"""Sorry, I couldn't find information about "{service_name}" in our database.

General Information:
You can contact the nearest Divisional Secretariat or visit www.gov.lk for comprehensive service information.

Available Services:
Try asking about: passport, driving license, birth certificate, marriage certificate, business registration, etc."""
            
            db_result = {"error": "Service not found"}
            return {**state, "response": response, "db_result": db_result, "service_name": service_name}
        
        service_info = service_response.data[0]
        
        # Get required documents for this service
        docs_response = client.table("required_doc_for_service").select(
            "doc_type_id, is_mandatory, document_type(doc_type, description)"
        ).eq("service_id", service_info['service_id']).execute()
        
        # Build response
        response = f"""{service_info['title']}

Required Documents:
"""
        documents = []
        if docs_response.data:
            for i, doc_relation in enumerate(docs_response.data, 1):
                doc_type = doc_relation['document_type']['doc_type']
                is_mandatory = doc_relation['is_mandatory']
                mandatory_text = " (Mandatory)" if is_mandatory else " (Optional)"
                response += f"{i}. {doc_type}{mandatory_text}\n"
                documents.append(doc_type)
        else:
            response += "No specific documents found in database. Please contact the department for details.\n"
        
        # Add department information
        dept_info = service_info['department']
        response += f"""
Department: {dept_info['title']}
Contact: {dept_info['email']} | {dept_info['phone_no']}

Service Description: {service_info['description'] or 'Contact department for details'}

Tip: Make sure all documents are original or certified copies, and bring photocopies as well."""

        db_result = {
            "service_name": service_info['title'],
            "documents": documents,
            "department": dept_info['title'],
            "contact": {"email": dept_info['email'], "phone": dept_info['phone_no']}
        }
        
    except Exception as e:
        print(f"Database error: {e}")  # For debugging
        response = f"""Unable to retrieve service information at the moment.

Alternative Options:
• Visit your nearest Divisional Secretariat
• Call 1919 (Government Information Center)
• Visit www.gov.lk

Please try again later or contact the relevant department directly."""
        
        db_result = {"error": str(e)}
    
    return {**state, "response": response, "db_result": db_result, "service_name": service_name}

def check_document_status(state: AgentState) -> AgentState:
    """Check user's document verification status"""
    # In a real implementation, you'd authenticate the user first
    response = """Document Status Check

To check your document status, you'll need to:

1. Log into your account on our portal
2. Provide your NIC number for verification
3. Select the specific application you want to check

Quick Status Check:
• SMS: Send NIC to 1919
• Online: Visit www.gov.lk/status
• Call: 1919 (Government Helpline)

Security Note: For privacy, I cannot access personal document status without proper authentication."""
    
    db_result = {"status_check": "authentication_required"}
    
    return {**state, "response": response, "db_result": db_result}

def handle_appointments(state: AgentState) -> AgentState:
    """Handle appointment scheduling queries"""
    response = """Appointment Booking

To schedule an appointment:

1. Choose your service (passport, license, etc.)
2. Select preferred date and time
3. Provide required documents
4. Confirm your appointment

Booking Methods:
• Online: www.gov.lk/appointments
• Phone: Call relevant department
• Visit: Walk-in service centers

Business Hours: Monday-Friday, 8:30 AM - 4:30 PM

Before Your Appointment:
• Prepare all required documents
• Arrive 15 minutes early
• Bring your appointment confirmation"""
    
    db_result = {"appointment_info": "general_guidance"}
    
    return {**state, "response": response, "db_result": db_result}

def handle_complaints(state: AgentState) -> AgentState:
    """Handle complaint filing"""
    response = """File a Complaint

Online Complaint:
• Visit www.gov.lk/complaints
• Fill complaint form with details
• Upload supporting documents
• Track complaint status

Phone Complaint:
• Call 1919 (Government Helpline)
• Provide complaint details
• Get reference number

Written Complaint:
• Submit to relevant department
• Include all supporting documents
• Keep copies for your records

What to Include:
• Date and time of incident
• Department/officer involved
• Detailed description
• Supporting evidence
• Your contact information

Emergency Issues: Contact department head directly"""
    
    db_result = {"complaint_channels": ["online", "phone", "written"]}
    
    return {**state, "response": response, "db_result": db_result}

def provide_general_info(state: AgentState) -> AgentState:
    """Provide general information about government services"""
    response = """🇱🇰 Sri Lankan Government Services Portal

Popular Services:
• Passport applications
• Driving licenses
• Birth/Marriage certificates
• Business registrations
• Tax services
• Healthcare services

How I Can Help:
• Find required documents for services
• Guide you through processes
• Provide department contact information
• Explain service requirements

Quick Commands:
• "What documents do I need for [service]?"
• "How to apply for [service]?"
• "Contact information for [department]?"

General Helpline: 1919
Official Website: www.gov.lk"""
    
    return {**state, "response": response}

def handle_general_queries(state: AgentState) -> AgentState:
    """Handle general queries with controlled responses"""
    user_input = state["user_input"].lower()
    
    # Check for common government-related queries
    if any(word in user_input for word in ["hours", "timing", "open", "closed"]):
        response = """Government Office Hours

Standard Hours: Monday - Friday, 8:30 AM - 4:30 PM
Lunch Break: 12:00 PM - 1:00 PM
Weekends: Closed (except emergency services)
Public Holidays: Closed

24/7 Services:
• Emergency services: 119, 110
• Government Helpline: 1919"""
    
    elif any(word in user_input for word in ["location", "address", "where", "find"]):
        response = """Service Locations

Divisional Secretariats: Available in every division
District Offices: Available in each district
Provincial Offices: Available in every province

Find Nearest Office:
• Visit www.gov.lk/locations
• Call 1919 for directions
• Use Google Maps: "government office near me"

Transportation: Most offices accessible by public transport"""
    
    else:
        response = """Government Services Assistant

I'm here to help with Sri Lankan government services information.

I can help you with:
• Required documents for services
• Application processes
• Department contact information
• General service guidance

For specific queries, try:
• "What documents do I need for a passport?"
• "How to apply for driving license?"
• "Contact information for registrar office?"

Need more help? Call 1919 (Government Information Center)"""
    
    return {**state, "response": response}

# Enhanced Workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify", classify_intent)
workflow.add_node("service_requirements", get_service_requirements)
workflow.add_node("check_documents", check_document_status)
workflow.add_node("schedule_appointment", handle_appointments)
workflow.add_node("file_complaint", handle_complaints)
workflow.add_node("general_info", provide_general_info)
workflow.add_node("general", handle_general_queries)

workflow.set_entry_point("classify")

def route_intent(state: AgentState):
    return state["intent"]

workflow.add_conditional_edges(
    "classify",
    route_intent,
    {
        "service_requirements": "service_requirements",
        "check_documents": "check_documents",
        "schedule_appointment": "schedule_appointment",
        "file_complaint": "file_complaint",
        "general_info": "general_info",
        "general": "general"
    }
)

# Add edges to END
for node in ["service_requirements", "check_documents", "schedule_appointment", 
             "file_complaint", "general_info", "general"]:
    workflow.add_edge(node, END)

agent = workflow.compile()

# Enhanced Main execution
if __name__ == "__main__":
    print("🇱🇰 Sri Lankan Government Services Chatbot")
    print("=" * 50)
    print("Ask me about:")
    print("• Required documents for services")
    print("• Application processes")
    print("• Department information")
    print("• General government services")
    print("=" * 50)
    print("Type 'exit' to quit\n")
    
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        if not user_input:
            print("Bot: Please ask me something about government services.\n")
            continue
        
        try:
            # Store user input
            conversation_history.append(f"You: {user_input}")
            
            # Process with agent
            result = agent.invoke({
                "user_input": user_input,
                "intent": "",
                "response": "",
                "db_result": {},
                "service_name": None,
                "user_context": None
            })
            
            # Get and display response
            bot_response = result['response']
            conversation_history.append(f"Bot: {bot_response}")
            
            print(f"Bot: {bot_response}\n")
            
        except Exception as e:
            error_msg = "Sorry, I encountered an error. Please try again or call 1919 for assistance."
            print(f"Bot: {error_msg}\n")
            conversation_history.append(f"Bot: {error_msg}")

    # Save conversation log with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_log_{timestamp}.txt"
    
    with open(filename, "w", encoding='utf-8') as file:
        file.write("Sri Lankan Government Services - Conversation Log\n")
        file.write("=" * 50 + "\n")
        file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("=" * 50 + "\n\n")
        
        for message in conversation_history:
            file.write(f"{message}\n")
        
        file.write("\n" + "=" * 50 + "\n")
        file.write("End of Conversation")

    print(f"Conversation saved to {filename}")
    print("Thank you for using Sri Lankan Government Services! 🇱🇰")