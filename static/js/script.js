document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const summarizeBtn = document.getElementById('summarizeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const wordCount = document.getElementById('wordCount');

    function updateWordCount() {
        const words = inputText.value.trim().split(/\s+/).filter(word => word.length > 0);
        wordCount.textContent = `Words: ${words.length}`;
        
        if (words.length < 50) {
            wordCount.style.color = '#dc2626';
            summarizeBtn.disabled = true;
        } else {
            wordCount.style.color = '#6b7280';
            summarizeBtn.disabled = false;
        }
    }

    inputText.addEventListener('input', updateWordCount);

    summarizeBtn.addEventListener('click', async function() {
        const text = inputText.value.trim();
        
        if (text.length === 0) {
            showError('Please enter some text to summarize');
            return;
        }
        
        if (text.split(/\s+/).length < 50) {
            showError('Please enter at least 50 words for better summarization');
            return;
        }
        
        loadingDiv.style.display = 'block';
        errorDiv.style.display = 'none';
        outputText.textContent = '';
        summarizeBtn.disabled = true;
        
        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Something went wrong');
            }
            
            outputText.textContent = data.summary;
        } catch (error) {
            showError(error.message);
        } finally {
            loadingDiv.style.display = 'none';
            summarizeBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', function() {
        inputText.value = '';
        outputText.textContent = '';
        errorDiv.style.display = 'none';
        updateWordCount();
    });

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        outputText.textContent = '';
    }

    updateWordCount();
});
