document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const outputContent = document.getElementById('outputContent');
    const outputPlaceholder = document.getElementById('outputPlaceholder');
    const outputActions = document.getElementById('outputActions');
    const summarizeBtn = document.getElementById('summarizeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const exampleBtn = document.getElementById('exampleBtn');
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    const errorMessage = document.getElementById('errorMessage');
    const wordCount = document.getElementById('wordCount');
    const charCount = document.getElementById('charCount');
    const summaryStats = document.getElementById('summaryStats');
    const compressionRatio = document.getElementById('compressionRatio');
    const reductionPercent = document.getElementById('reductionPercent');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    // Example text
    const exampleText = `Text summarization is the process of creating a shorter version of a longer text while preserving the key information and meaning. This is useful for quickly understanding long documents, articles, or reports. Natural language processing techniques like transformer models have greatly improved the quality of automated text summarization.

These models can understand context, identify important information, and generate coherent summaries that capture the essence of the original text. The BART model used in this application has been fine-tuned on dialogue data to better understand conversational context and generate meaningful summaries.

The application processes text through several stages: tokenization, encoding, attention mechanisms, and decoding. The model considers the entire input text, assigns importance to different parts, and generates a summary that maintains factual accuracy while being concise and readable.

This technology has numerous applications in education, business, journalism, and research, helping people save time and focus on key information.`;
    
    // Initialize
    updateCounts();
    checkSystemStatus();
    
    // Event Listeners
    inputText.addEventListener('input', updateCounts);
    
    summarizeBtn.addEventListener('click', summarizeText);
    
    clearBtn.addEventListener('click', function() {
        inputText.value = '';
        clearOutput();
        updateCounts();
    });
    
    exampleBtn.addEventListener('click', function() {
        inputText.value = exampleText;
        updateCounts();
        inputText.focus();
    });
    
    copyBtn.addEventListener('click', copySummary);
    
    downloadBtn.addEventListener('click', downloadSummary);
    
    // Update word and character counts
    function updateCounts() {
        const text = inputText.value.trim();
        const words = text ? text.split(/\s+/).filter(word => word.length > 0) : [];
        const characters = text.length;
        
        wordCount.querySelector('span').textContent = words.length;
        charCount.querySelector('span').textContent = characters;
        
        // Update button state
        if (words.length >= 30 && words.length <= 2000) {
            summarizeBtn.disabled = false;
            summarizeBtn.innerHTML = '<i class="fas fa-magic"></i> Generate Summary';
        } else {
            summarizeBtn.disabled = true;
            if (words.length < 30) {
                summarizeBtn.innerHTML = `<i class="fas fa-info-circle"></i> Need ${30 - words.length} more words`;
            } else {
                summarizeBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Text too long';
            }
        }
    }
    
    // Summarize text
    async function summarizeText() {
        const text = inputText.value.trim();
        
        if (!text) {
            showError('Please enter some text to summarize');
            return;
        }
        
        const words = text.split(/\s+/).filter(word => word.length > 0);
        if (words.length < 30) {
            showError('Please enter at least 30 words for better summarization');
            return;
        }
        
        if (words.length > 2000) {
            showError('Text is too long. Please limit to 2000 words');
            return;
        }
        
        // Show loading state
        showLoading();
        hideError();
        clearOutput();
        
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
                throw new Error(data.error || 'Failed to generate summary');
            }
            
            // Display summary
            displaySummary(data.summary, data.stats);
            
        } catch (error) {
            showError(error.message);
        } finally {
            hideLoading();
        }
    }
    
    // Display summary
    function displaySummary(summary, stats) {
        outputText.textContent = summary;
        outputPlaceholder.style.display = 'none';
        outputContent.style.display = 'block';
        outputActions.style.display = 'flex';
        
        // Update stats
        if (stats) {
            summaryStats.style.display = 'flex';
            compressionRatio.textContent = stats.compression_ratio;
            reductionPercent.textContent = stats.reduction_percentage;
        }
        
        // Scroll to output
        outputContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Clear output
    function clearOutput() {
        outputText.textContent = '';
        outputPlaceholder.style.display = 'flex';
        outputContent.style.display = 'none';
        outputActions.style.display = 'none';
        summaryStats.style.display = 'none';
        hideError();
    }
    
    // Show loading state
    function showLoading() {
        loadingState.style.display = 'block';
        summarizeBtn.disabled = true;
        summarizeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
    
    // Hide loading state
    function hideLoading() {
        loadingState.style.display = 'none';
        summarizeBtn.disabled = false;
        updateCounts();
    }
    
    // Show error
    function showError(message) {
        errorMessage.textContent = message;
        errorState.style.display = 'block';
        errorState.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    // Hide error
    function hideError() {
        errorState.style.display = 'none';
    }
    
    // Copy summary to clipboard
    async function copySummary() {
        const summary = outputText.textContent;
        
        try {
            await navigator.clipboard.writeText(summary);
            
            // Visual feedback
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyBtn.classList.add('btn-success');
            
            setTimeout(() => {
                copyBtn.innerHTML = originalText;
                copyBtn.classList.remove('btn-success');
            }, 2000);
            
        } catch (err) {
            showError('Failed to copy to clipboard');
        }
    }
    
    // Download summary as text file
    function downloadSummary() {
        const summary = outputText.textContent;
        const originalWords = inputText.value.split(/\s+/).length;
        const summaryWords = summary.split(/\s+/).length;
        
        const content = `AI Generated Summary\n` +
                       `=====================\n\n` +
                       `${summary}\n\n` +
                       `=====================\n` +
                       `Original: ${originalWords} words\n` +
                       `Summary: ${summaryWords} words\n` +
                       `Compression: ${compressionRatio.textContent}x\n` +
                       `Generated by: Text Summarization AI\n` +
                       `Date: ${new Date().toLocaleString()}`;
        
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `summary_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // Check system status
    async function checkSystemStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                statusIndicator.style.background = '#4ade80';
                statusText.textContent = 'System Online';
                statusIndicator.style.animation = 'none';
            } else {
                statusIndicator.style.background = '#f59e0b';
                statusText.textContent = 'System Degraded';
            }
        } catch (error) {
            statusIndicator.style.background = '#ef4444';
            statusText.textContent = 'System Offline';
            statusIndicator.style.animation = 'none';
        }
    }
    
    // Auto-check status every 30 seconds
    setInterval(checkSystemStatus, 30000);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + Enter to summarize
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            if (!summarizeBtn.disabled) {
                summarizeText();
            }
        }
        
        // Escape to clear
        if (event.key === 'Escape') {
            clearBtn.click();
        }
    });
});
