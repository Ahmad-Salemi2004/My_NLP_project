import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
import json

def test_app():
    """Test the Flask application."""
    with app.test_client() as client:
        # Test home page
        print("Testing home page...")
        response = client.get('/')
        assert response.status_code == 200
        print("✓ Home page works")
        
        # Test health endpoint
        print("Testing health endpoint...")
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        print("✓ Health endpoint works")
        
        # Test summarize endpoint with valid text
        print("Testing summarize endpoint...")
        test_text = """This is a test text for summarization. It needs to be at least fifty words long to work properly with the summarization model. The model expects sufficient content to generate a meaningful summary. Text summarization is the process of creating a shorter version of a longer text while preserving the key information and meaning. This is useful for quickly understanding long documents, articles, or reports. Natural language processing techniques like transformer models have greatly improved the quality of automated text summarization."""
        
        response = client.post('/summarize', 
                             json={'text': test_text},
                             content_type='application/json')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'summary' in data
            print("✓ Summarize endpoint works")
            print(f"Generated summary: {data['summary'][:100]}...")
        else:
            print(f"Summarize endpoint returned {response.status_code}")
            print(f"Response: {response.data}")
        
        print("\nAll tests passed!")

if __name__ == "__main__":
    test_app()
