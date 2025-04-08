(function(){
    let selectedFile = null;

    function previewFile(file) {
        const previewPlaceholder = document.getElementById('preview-placeholder');
        const imagePreview = document.getElementById('preview-area');
        const pdfPreview = document.getElementById('pdf-preview');
        const fileInfo = document.getElementById('file-info');

        // Reset all previews
        previewPlaceholder.classList.remove('hidden');
        imagePreview.classList.add('hidden');
        pdfPreview.classList.add('hidden');

        const reader = new FileReader();
        reader.onload = (e) => {
            if (file.type.startsWith('image/')) {
                // Image preview
                imagePreview.src = e.target.result;
                previewPlaceholder.classList.add('hidden');
                imagePreview.classList.remove('hidden');
            } else if (file.type === 'application/pdf') {
                // PDF Preview
                const pdfURL = URL.createObjectURL(file);
                pdfPreview.src = pdfURL;
                
                pdfPreview.onload = () => {
                    previewPlaceholder.classList.add('hidden');
                    imagePreview.classList.add('hidden');
                    pdfPreview.classList.remove('hidden');
                };

                pdfPreview.onerror = () => {
                    console.warn("Native PDF preview failed.");
                    previewPlaceholder.innerHTML = `
                        <i class="fas fa-file-pdf text-6xl text-red-500 mb-4"></i>
                        <p>無法預覽PDF文件</p>
                    `;
                    previewPlaceholder.classList.remove('hidden');
                    pdfPreview.classList.add('hidden');
                };
            }
        };

        reader.readAsDataURL(file);
    }

    async function performOCR(file) {
        const formData = new FormData();
        formData.append("file", file);
    
        try {
            const response = await fetch("/perform_ocr", {
                method: "POST",
                body: formData
            });
    
            if (!response.ok) {
                throw new Error("OCR處理失敗");
            }
    
            const result = await response.json();
            
            if (result.success) {
                // 显示识别文本
                document.getElementById('ocr-result').textContent = result.text;
                alert("OCR處理完成");
            } else {
                alert("未能識別任何文字");
            }
        } catch (error) {
            console.error("OCR Error:", error);
            alert("OCR處理過程發生錯誤");
        }
    }

    function initializeOCRInterface() {
        const fileInput = document.getElementById('file-input');
        const browseBtn = document.getElementById('browse-btn');
        const doOcrBtn = document.getElementById('do-ocr-btn');
        const fileInfo = document.getElementById('file-info');

        // File Browse Button
        browseBtn.addEventListener('click', () => {
            fileInput.click();
        });

        // File Selection
        fileInput.addEventListener('change', (event) => {
            selectedFile = event.target.files[0];
            if (selectedFile) {
                fileInfo.textContent = `已選擇文件: ${selectedFile.name}`;
                
                // Preview the selected file
                previewFile(selectedFile);
            } else {
                fileInfo.textContent = '尚未選擇文件';
            }
        });

        // OCR Button
        doOcrBtn.addEventListener('click', () => {
            if (!selectedFile) {
                alert('請先選擇文件!');
                return;
            }
            
            // Perform OCR on the selected file
            performOCR(selectedFile);
        });
    }

    // Initialize the interface when the script loads
    window.initializeOCRInterface = initializeOCRInterface;
})();
