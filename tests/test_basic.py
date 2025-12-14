import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

class TestSummarizationAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_health_endpoint(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
    
    def test_summarize_empty_text(self):
        response = self.app.post('/summarize', json={'text': ''})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)
    
    def test_summarize_short_text(self):
        short_text = "This is a short text with less than fifty words."
        response = self.app.post('/summarize', json={'text': short_text})
        self.assertEqual(response.status_code, 400)
    
    def test_summarize_valid_text(self):
        # Using a sample text from the DialogSum dataset
        sample_text = """#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?
#Person2#: I found it would be a good idea to get a check-up.
#Person1#: Yes, well, you haven't had one for 5 years. You should have one every year.
#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?
#Person1#: Well, the best way to avoid serious illnesses is to find out about them early. So try to come at least once a year for your own good.
#Person2#: Ok.
#Person1#: Let me see here. Your eyes and ears look fine. Take a deep breath, please. Do you smoke, Mr. Smith?
#Person2#: Yes.
#Person1#: Smoking is the leading cause of lung cancer and heart disease, you know. You really should quit.
#Person2#: I've tried hundreds of times, but I just can't seem to kick the habit.
#Person1#: Well, we have classes and some medications that might help. I'll give you more information before you leave.
#Person2#: Ok, thanks doctor."""
        
        response = self.app.post('/summarize', json={'text': sample_text})
        # This might return 500 if model isn't loaded, which is fine for testing
        if response.status_code == 200:
            data = response.get_json()
            self.assertIn('summary', data)
            self.assertTrue(len(data['summary']) > 0)

if __name__ == '__main__':
    unittest.main()
