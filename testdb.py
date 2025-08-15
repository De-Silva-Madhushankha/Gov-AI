from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase_client() -> Client:
    """Get Supabase client connection"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return None

def test_supabase_connection():
    client = get_supabase_client()
    if client is None:
        print("Supabase connection failed.")
        return False
    try:
        response = client.table("service").select("*").limit(1).execute()
        if response.data:
            print("Supabase connection successful.")
            return True
        else:
            print("Connected, but no data found in 'Service' table.")
            return True
    except Exception as e:
        print(f"Supabase query failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Supabase connection...")
    test_supabase_connection()