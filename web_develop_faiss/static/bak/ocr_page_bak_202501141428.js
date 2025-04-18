(function(){
    
    let selectedFile = null;

    // 定義初始化聊天介面的函數
    function initializeOCRInterface() {
        console.log("初始化OCR介面...");

        // 確保 DOM 元素存在
       // 文件選擇和上傳邏輯
        const fileInput = document.getElementById('file-input');
        const browseBtn = document.getElementById('browse-btn');
        const uploadBtn = document.getElementById('upload-btn');
        const fileInfo = document.getElementById('file-info');
        const progressBar = document.getElementById('progress');
        const doOcrBtn = document.getElementById('do-ocr-btn');

        if (!fileInput || !browseBtn || !uploadBtn || !fileInfo || !progressBar || !doOcrBtn) {
            console.error("OCR介面的必要 DOM 元素未找到，稍後重試...");
            return;
        }

        // 點擊 "Browse" 按鈕，觸發文件選擇
        browseBtn.addEventListener('click', () => {
            fileInput.click();
        });
  
      // 文件選擇後，顯示文件名
      fileInput.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
          fileInfo.textContent = `Selected file: ${selectedFile.name}`;
        } else {
          fileInfo.textContent = 'No file selected';
        }
      });

      // 點擊 "Upload" 按鈕，上傳文件
        uploadBtn.addEventListener("click", async () => {
            if (!selectedFile) {
                alert("Please select a file first!");
                return;
            }

            const formData = new FormData();
            formData.append("file", selectedFile);

            try {
                const response = await fetch("/upload_file", {
                    method: "POST",
                    body: formData,
                });

                if (response.ok) {
                    const result = await response.json();
                    fileInfo.textContent = `Uploaded file: ${result.filename}`;
                    updateStatusMessage("✅ File uploaded successfully!");
                    progressBar.style.width = "100%";
                } else {
                    throw new Error("File upload failed");
                }
            } catch (error) {
                console.error(error);
                updateStatusMessage("❌ Error uploading file");
            }
        });
  
  
      // 點擊 "Do OCR" 按鈕，檢查文件並執行 OCR
        doOcrBtn.addEventListener('click', () => {
            if (!selectedFile) {
            alert('Please upload a file first!');
            return;
            }
    
            alert(`Starting OCR for file: ${selectedFile.name}`);
            // 在這裡調用後端的 OCR API
        });

        fileInput.addEventListener('change', handleInputChange);

        function updateStatusMessage(text) {
            statusMessage.textContent = text;
        }

        function assertFilesValid(fileList) {
            const allowedTypes = ['image/jpeg', 'image/png','application/pdf'];
            const sizeLimit = 10 * 1024 * 1024; // 1 megabyte

            // const { name: fileName, size: fileSize } = uploadfile;
            // if (!allowedTypes.includes(uploadfile.type)) {
            //     throw new Error(`❌ File "${fileName}" could not be uploaded. Only images with the following types are allowed: WEBP, JPEG, PNG.`);
            // }
            for (const file of fileList) {
                const { name: fileName, size: fileSize } = file;
            
                if (!allowedTypes.includes(file.type)) {
                    throw new Error(`❌ File "${fileName}" 無法上傳. 只有圖片及PDF文件可以上傳。允許的圖片格式: JPEG, PNG.`);
                }
                // ↓ the new condition ↓
               if (fileSize > sizeLimit) {
                 throw new Error(`❌ File "${fileName}" could not be uploaded. Only images up to 10 MB are allowed.`);
               }
            }
        }

        function resetFormState() {
            uploadBtn.disabled = true;
            updateStatusMessage(`🤷‍♂ Nothing's uploaded`)
        }

        function handleInputChange() {
            try {
                resetFormState()
                assertFilesValid(fileInput.files);
            } catch (err) {
            updateStatusMessage(err.message);
            return;
            }
            uploadBtn.disabled = false;
        }

    }

    // 將函數掛載到全局作用域
    window.initializeOCRInterface = initializeOCRInterface;

    

    
})();
