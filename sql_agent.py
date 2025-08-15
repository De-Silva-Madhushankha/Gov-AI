from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict, Any
import json
from datetime import datetime
from supabase import create_client, Client
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class AgentState(TypedDict):
    user_input: str
    intent: str
    response: str
    sql_query: str
    sql_result: List[Dict[Any, Any]]
    error: Optional[str]
    needs_clarification: bool

try:
    llm = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    llm = None

# Database schema for the AI to understand
DATABASE_SCHEMA = """
Database Schema:
================

Tables:
1. app_user (user_id, full_name, nic_no, email, phone_no, password, role, created_at, updated_at)
2. department (department_id, title, description, email, phone_no, created_at, updated_at)
3. service (service_id, department_id, title, description, created_at, updated_at)
4. document_type (doc_type_id, doc_type, description, created_at)
5. required_doc_for_service (service_id, doc_type_id, is_mandatory, created_at)
6. appointment (appointment_id, officer_id, citizen_id, service_id, timeslot_id, status, created_at, updated_at)
7. time_slot (timeslot_id, service_id, start_time, end_time, max_appointments, created_at)
8. user_document (user_doc_id, user_id, doc_type_id, file_path, upload_time, verification_status)
9. appointment_document (appointment_doc_id, appointment_id, file_path, doc_type, uploaded_date, verification_status, review)
10. feedback (feedback_id, appointment_id, rating, review, submit_time)
11. complaint (complaint_id, citizen_id, appointment_id, title, field, type, created_at)
12. notification (notification_id, appointment_id, citizen_id, type, message, send_via, send_time, created_at)
13. message (message_id, appointment_id, sender_id, receiver_id, message, send_time)
14. officer_department (officer_id, department_id, created_at)
15. user_auth (user_id, auth_user_id)

Key Relationships:
- service.department_id -> department.department_id
- required_doc_for_service.service_id -> service.service_id
- required_doc_for_service.doc_type_id -> document_type.doc_type_id
- appointment.citizen_id -> app_user.user_id
- appointment.officer_id -> app_user.user_id
- appointment.service_id -> service.service_id
"""

def get_supabase_client() -> Client:
    """Get Supabase client connection"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except Exception as e:
        return None

def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent and determine if SQL query is needed"""
    user_input = state["user_input"]
    
    try:
        prompt = f"""
        Analyze the following user query about Sri Lankan government services and classify it:

        Categories:
        - sql_query: User is asking for specific data that requires database lookup (documents for services, service details, department info, etc.)
        - procedural_info: User asking about processes, general guidance, how-to information
        - status_check: User wants to check application/document status (requires authentication)
        - general: General questions about government services

        User query: "{user_input}"

        Examples of sql_query:
        - "What documents do I need for passport?"
        - "Which services does Health Department offer?"
        - "Show me all available services"
        - "What are the requirements for driving license?"

        Examples of procedural_info:
        - "How do I apply for passport?"
        - "What are the office hours?"
        - "How to file a complaint?"

        Respond with only the category name.
        """
        
        response = llm.generate_content(prompt)
        intent = response.text.strip().lower()
        
        valid_intents = ["sql_query", "procedural_info", "status_check", "general"]
        if intent not in valid_intents:
            intent = "general"
            
    except Exception as e:
        intent = "general"
    
    return {**state, "intent": intent}

def generate_sql_query(state: AgentState) -> AgentState:
    """Generate SQL query based on user input using AI"""
    user_input = state["user_input"]
    
    try:
        prompt = f"""
        Based on the user question and database schema, generate a PostgreSQL query.

        {DATABASE_SCHEMA}

        User Question: "{user_input}"

        Rules:
        1. Use PostgreSQL syntax
        2. Use JOINs to get related data when needed
        3. Use ILIKE for case-insensitive pattern matching
        4. Limit results to reasonable numbers (use LIMIT)
        5. Only SELECT queries are allowed
        6. Focus on public data only
        7. Use table aliases for readability

        Common patterns:
        - For service requirements: JOIN service, department, required_doc_for_service, document_type
        - For department services: JOIN department and service tables
        - For document types: Use document_type table

        Example queries:
        - Documents for passport: 
          SELECT s.title as service_name, dt.doc_type, rds.is_mandatory, d.title as department
          FROM service s 
          JOIN department d ON s.department_id = d.department_id
          JOIN required_doc_for_service rds ON s.service_id = rds.service_id
          JOIN document_type dt ON rds.doc_type_id = dt.doc_type_id
          WHERE s.title ILIKE '%passport%';

        Generate only the SQL query, no explanations:
        """
        
        response = llm.generate_content(prompt)
        sql_query = response.text.strip()
        
        # Clean the SQL query
        sql_query = re.sub(r'^```sql\s*', '', sql_query)
        sql_query = re.sub(r'^```\s*', '', sql_query)
        sql_query = re.sub(r'```\s*$', '', sql_query)
        sql_query = sql_query.strip()
        
        # Basic security check - only allow SELECT statements
        if not sql_query.upper().startswith('SELECT'):
            return {**state, "error": "Invalid query type", "sql_query": ""}
        
        # Prevent dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            return {**state, "error": "Query contains forbidden operations", "sql_query": ""}
        
    except Exception as e:
        return {**state, "error": "Unable to process your request", "sql_query": ""}
    
    return {**state, "sql_query": sql_query, "error": None}

def execute_sql_query(state: AgentState) -> AgentState:
    """Execute the generated SQL query"""
    sql_query = state["sql_query"]
    
    if not sql_query or state.get("error"):
        return state
    
    try:
        client = get_supabase_client()
        if client is None:
            return {**state, "error": "Database connection failed", "sql_result": []}
        
        # Parse query and try to execute with Supabase client methods
        sql_result = execute_with_supabase_client(client, sql_query)
        
    except Exception as e:
        return {**state, "error": "Unable to retrieve information at this time", "sql_result": []}
    
    return {**state, "sql_result": sql_result, "error": None}

def execute_with_supabase_client(client: Client, sql_query: str) -> List[Dict]:
    """Fallback method to execute common queries using Supabase client"""
    sql_lower = sql_query.lower()
    
    try:
        # Pattern matching for common queries
        if 'service' in sql_lower and 'document_type' in sql_lower:
            # Service requirements query
            service_match = re.search(r"title\s+ilike\s+'%([^%]+)%'", sql_lower)
            if service_match:
                service_name = service_match.group(1)
                result = client.table("service").select(
                    "title, description, department(title, email), required_doc_for_service(is_mandatory, document_type(doc_type, description))"
                ).ilike("title", f"%{service_name}%").execute()
                return result.data
        
        elif 'service' in sql_lower and 'department' in sql_lower:
            # Department services query
            dept_match = re.search(r"title\s+ilike\s+'%([^%]+)%'", sql_lower)
            if dept_match:
                dept_name = dept_match.group(1)
                result = client.table("service").select(
                    "title, description, department!inner(title, email, phone_no)"
                ).filter("department.title", "ilike", f"%{dept_name}%").execute()
                return result.data
            else:
                result = client.table("service").select(
                    "title, description, department(title, email, phone_no)"
                ).limit(20).execute()
                return result.data
        
        elif 'document_type' in sql_lower:
            # Document types query
            result = client.table("document_type").select("doc_type, description").limit(20).execute()
            return result.data
        
        elif 'department' in sql_lower:
            # Department query
            result = client.table("department").select("title, description, email, phone_no").limit(20).execute()
            return result.data
        
        else:
            # Default: try to get services
            result = client.table("service").select("title, description").limit(10).execute()
            return result.data
            
    except Exception as e:
        return []

def format_sql_response(state: AgentState) -> AgentState:
    """Format the SQL query results into a user-friendly response"""
    sql_result = state["sql_result"]
    user_input = state["user_input"]
    error = state.get("error")
    
    if error:
        response = f"""I'm sorry, I couldn't process your request at the moment.

You can try:
• Rephrasing your question in a different way
• Being more specific about what you're looking for
• Contacting our helpline at 1919 for immediate assistance

Alternative ways to get help:
• Visit your nearest Divisional Secretariat office
• Check our official website at www.gov.lk
• Call the Government Information Center at 1919"""
        return {**state, "response": response}
    
    if not sql_result:
        response = """I couldn't find specific information for your query.

Try asking about:
• "What documents do I need for passport application?"
• "What services does the Health Department offer?"
• "Show me all available government services"
• "What are the requirements for driving license?"

Need immediate help?
• Call 1919 (Government Information Center)
• Visit www.gov.lk for comprehensive information"""
        return {**state, "response": response}
    
    try:
        # Use AI to format the results nicely
        prompt = f"""
        Format this database query result into a user-friendly response for: "{user_input}"

        Query Results:
        {json.dumps(sql_result, indent=2)}

        Instructions:
        1. Create a clear, well-organized response
        2. Use bullet points or numbered lists for easy reading
        3. Include contact information (email, phone) when available
        4. Add helpful tips or next steps for the user
        5. Keep it concise but informative
        6. Use appropriate emojis to make it more engaging
        7. Format as plain text (not markdown)
        8. If showing document requirements, clearly mark mandatory vs optional
        9. Include department information when relevant
        10. End with helpful contact information or next steps

        Make it sound professional but friendly, like a helpful government service representative.
        """
        
        ai_response = llm.generate_content(prompt)
        response = ai_response.text.strip()
        
    except Exception as e:
        # Enhanced fallback formatting
        response = "Here's what I found for you:\n\n"
        
        for i, item in enumerate(sql_result[:5], 1):
            response += f"{i}. "
            if isinstance(item, dict):
                # Smart formatting based on data type
                if 'title' in item:
                    response += f"{item['title']}\n"
                
                if 'description' in item and item['description']:
                    response += f"   Description: {item['description']}\n"
                
                if 'doc_type' in item:
                    mandatory = item.get('is_mandatory', True)
                    status = " (Required)" if mandatory else " (Optional)"
                    response += f"   Document: {item['doc_type']}{status}\n"
                
                if 'department' in item and isinstance(item['department'], dict):
                    dept = item['department']
                    response += f"   Department: {dept.get('title', 'N/A')}\n"
                    if dept.get('email'):
                        response += f"   Contact: {dept['email']}"
                    if dept.get('phone_no'):
                        response += f" | {dept['phone_no']}"
                    response += "\n"
                
                if 'email' in item and item['email']:
                    response += f"   Email: {item['email']}\n"
                
                if 'phone_no' in item and item['phone_no']:
                    response += f"   Phone: {item['phone_no']}\n"
                
                response += "\n"
        
        if len(sql_result) > 5:
            response += f"... and {len(sql_result) - 5} more results available.\n\n"
        
        response += """Need more information?
• Call 1919 (Government Information Center)
• Visit www.gov.lk
• Contact the relevant department directly"""
    
    return {**state, "response": response}

def handle_procedural_info(state: AgentState) -> AgentState:
    """Handle procedural and general information queries"""
    user_input = state["user_input"].lower()
    
    if "office hours" in user_input or "timing" in user_input or "time" in user_input:
        response = """Government Office Hours

Regular Working Hours:
• Monday - Friday: 8:30 AM - 4:30 PM
• Lunch Break: 12:00 PM - 1:00 PM
• Weekends: Closed (except emergency services)
• Public Holidays: Closed

24/7 Emergency Services:
• Police Emergency: 119
• Fire & Rescue: 110
• Government Helpline: 1919

Tip: Call ahead to confirm specific department hours as some may vary."""
    
    elif "complaint" in user_input or "file complaint" in user_input:
        response = """How to File a Complaint

Online Complaint:
• Visit: www.gov.lk/complaints
• Fill out the complaint form with detailed information
• Upload any supporting documents
• Get a reference number to track your complaint

Phone Complaint:
• Call: 1919 (Government Helpline)
• Provide all relevant details
• Note down the reference number given

In-Person Complaint:
• Visit the relevant department office
• Bring all supporting documents
• Submit written complaint with copies

Information to Include:
• Date and time of incident
• Department/officer involved
• Detailed description of the issue
• Your contact information
• Any supporting evidence

Follow-up: You can track your complaint status using the reference number provided."""
    
    elif "apply" in user_input or "application" in user_input:
        response = """General Application Process

Step-by-Step Process:
1. Check required documents for your service
2. Prepare all necessary paperwork
3. Submit your application (online or in-person)
4. Pay applicable fees
5. Receive acknowledgment receipt
6. Track your application status

Online Applications:
• Visit: www.gov.lk
• Create an account or log in
• Select your required service
• Upload required documents
• Pay fees online

In-Person Applications:
• Visit the relevant department office
• Bring original documents and photocopies
• Submit application with fees
• Get receipt for tracking

Need Help?
• Call 1919 for guidance
• Visit your nearest Divisional Secretariat"""
    
    else:
        response = """Welcome to Sri Lankan Government Services

How I Can Help You:
• Find required documents for any government service
• Get department contact information
• Learn about service requirements and processes
• Understand application procedures

Try Asking Me:
• "What documents do I need for passport?"
• "Which services does the Health Department offer?"
• "How do I apply for a driving license?"
• "What are the office hours?"

Quick Help:
• Government Helpline: 1919
• Official Website: www.gov.lk
• Emergency Services: 119 (Police), 110 (Fire)

Find Your Nearest Office:
Visit www.gov.lk/locations or call 1919 for directions to the nearest government service center."""
    
    return {**state, "response": response}

def handle_status_check(state: AgentState) -> AgentState:
    """Handle status check requests"""
    response = """Check Your Application Status

For Security Reasons, Status Checks Require Authentication

Online Status Check:
• Visit: www.gov.lk/status
• Log in with your credentials
• Enter your application reference number
• View real-time status updates

SMS Status Check:
• Send your NIC number to 1919
• Receive instant status update
• Available 24/7

Phone Status Check:
• Call: 1919 (Government Information Center)
• Provide your NIC number and application reference
• Speak with a customer service representative
• Available: Monday-Friday, 8:30 AM - 4:30 PM

In-Person Status Check:
• Visit the department where you submitted your application
• Bring your NIC and application receipt
• Get detailed status information from the officer

What You'll Need:
• Your NIC number
• Application reference number
• Original receipt (for in-person visits)

Tip: Keep your application reference number safe - you'll need it for all status inquiries."""
    
    return {**state, "response": response}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify", classify_intent)
workflow.add_node("generate_sql", generate_sql_query)
workflow.add_node("execute_sql", execute_sql_query)
workflow.add_node("format_response", format_sql_response)
workflow.add_node("procedural_info", handle_procedural_info)
workflow.add_node("status_check", handle_status_check)

workflow.set_entry_point("classify")

def route_intent(state: AgentState):
    return state["intent"]

workflow.add_conditional_edges(
    "classify",
    route_intent,
    {
        "sql_query": "generate_sql",
        "procedural_info": "procedural_info",
        "status_check": "status_check",
        "general": "procedural_info"
    }
)

# SQL query flow
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "format_response")

# End points
workflow.add_edge("format_response", END)
workflow.add_edge("procedural_info", END)
workflow.add_edge("status_check", END)

agent = workflow.compile()

# Main execution
if __name__ == "__main__":
    print("Sri Lankan Government Services Assistant")
    print("=" * 55)
    print("🤖 Hello! I'm here to help you with government services.")
    print("I can help you find:")
    print("   • Required documents for services")
    print("   • Department contact information")
    print("   • Service procedures and requirements")
    print("   • Office hours and locations")
    print("=" * 55)
    print("Just ask me your question in plain English!")
    print("Type 'exit' when you're done.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\nThank you for using Sri Lankan Government Services Assistant!")
            print("Remember, you can always call 1919 for immediate assistance.")
            print("Visit www.gov.lk for online services.")
            print("Have a great day! 🇱🇰")
            break
        
        if not user_input:
            print("🤖 Assistant: I'm here to help! Please ask me about any government service you need information about.\n")
            continue
        
        try:
            print("Looking up information for you...")
            
            result = agent.invoke({
                "user_input": user_input,
                "intent": "",
                "response": "",
                "sql_query": "",
                "sql_result": [],
                "error": None,
                "needs_clarification": False
            })
            
            bot_response = result['response']
            print(f"\n🤖 Assistant: {bot_response}\n")
            print("-" * 55)
            
        except Exception as e:
            print(f"\n🤖 Assistant: I apologize, but I'm experiencing technical difficulties right now.")
            print("Please try again in a moment, or call 1919 for immediate assistance.")
            print("You can also visit www.gov.lk for online services.\n")
            print("-" * 55)