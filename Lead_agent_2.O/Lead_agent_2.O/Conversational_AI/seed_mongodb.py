"""
Script to seed MongoDB with question sets from JSON file
"""
import json
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os

load_dotenv()


def convert_oid(data):
    """Recursively convert $oid strings to ObjectId instances"""
    if isinstance(data, dict):
        if '$oid' in data:
            return ObjectId(data['$oid'])
        return {key: convert_oid(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_oid(item) for item in data]
    return data


def seed_mongodb():
    """Load question sets from JSON file and insert into MongoDB"""
    
    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    collection = db["llm_questions"]
    
    # Load JSON file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, "chatbot-ai.llm_questions.json")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            question_sets = json.load(f)
        
        # Convert $oid to ObjectId
        question_sets = convert_oid(question_sets)
        
        # Clear existing data (optional - comment out if you want to keep existing data)
        # collection.delete_many({})
        
        # Insert question sets
        if isinstance(question_sets, list):
            result = collection.insert_many(question_sets)
            print(f"✅ Successfully inserted {len(result.inserted_ids)} question set(s)")
        else:
            result = collection.insert_one(question_sets)
            print(f"✅ Successfully inserted 1 question set")
        
        # Verify insertion
        count = collection.count_documents({})
        print(f"✅ Total question sets in database: {count}")
        
        # List all question set titles
        titles = collection.find({}, {"title": 1, "_id": 0})
        print("\n📋 Question Sets:")
        for idx, doc in enumerate(titles, 1):
            print(f"   {idx}. {doc.get('title', 'N/A')}")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{json_file}' not found")
        print("   Please ensure chatbot-ai.llm_questions.json exists in the project root")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {json_file}: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    print("🌱 Seeding MongoDB with question sets...")
    print("=" * 50)
    seed_mongodb()
    print("=" * 50)
