document.addEventListener('DOMContentLoaded', function() {
    // Handle image upload form submission
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('image');
            formData.append('file', fileInput.files[0]);
            
            // Show loading state
            const resultsDiv = document.getElementById('recognitionResults');
            resultsDiv.innerHTML = '<div class="text-center">Processing image...</div>';
            
            // Send the image to the server
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Display the uploaded image
                const uploadedImageDiv = document.getElementById('uploadedImage');
                uploadedImageDiv.innerHTML = `<img src="data:image/jpeg;base64,${data.image}" alt="Uploaded Image">`;
                
                // Display recognition results
                let resultsHtml = '<h6>Recognition Results:</h6>';
                if (data.faces && data.faces.length > 0) {
                    data.faces.forEach(face => {
                        resultsHtml += `
                            <div class="mb-2">
                                <strong>Name:</strong> ${face.name}<br>
                                <strong>Confidence:</strong> ${(face.confidence * 100).toFixed(2)}%
                            </div>
                        `;
                    });
                } else {
                    resultsHtml += '<p>No faces detected in the image.</p>';
                }
                resultsDiv.innerHTML = resultsHtml;
            })
            .catch(error => {
                console.error('Error:', error);
                resultsDiv.innerHTML = '<div class="alert alert-danger">An error occurred while processing the image.</div>';
            });
        });
    }
    
    // Preview image before upload
    const imageInput = document.getElementById('image');
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const uploadedImageDiv = document.getElementById('uploadedImage');
                    uploadedImageDiv.innerHTML = `<img src="${e.target.result}" alt="Preview" class="img-fluid">`;
                };
                reader.readAsDataURL(file);
            }
        });
    }
}); 