import express from 'express';
import fs from 'fs';
import path from 'path';
import { pipeline } from 'stream';
import { exec } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Directory to save uploaded files
const uploadDir = path.join(__dirname, 'uploads');

// Ensure the upload directory exists
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

// Endpoint to handle file uploads
app.post('/', (req, res) => {
    // Check content type
    const contentType = req.headers['content-type'];
    if (!contentType || (!contentType.includes('image/jpeg') && !contentType.includes('image/png'))) {
        return res.status(400).send('Only JPG and PNG files are allowed.');
    }

    // Generate unique file name
    const ext = contentType.includes('image/jpeg') ? 'jpg' : 'png';
    const fileName = `${Date.now()}.${ext}`;
    const filePath = path.join(uploadDir, fileName);

    // Save the stream to the file system
    const writeStream = fs.createWriteStream(filePath);

    pipeline(req, writeStream, (err) => {
        if (err) {
            console.error('File upload error:', err);
            return res.status(500).send('File upload failed.');
        }
        console.log(`File saved as ${fileName}`);
        // Execute OCR
        const tesseract = exec(`tesseract "${filePath}" - -l eng+fra`, (error, stdout, stderr) => {
            if (error) {
                console.log(`error: ${error.message}`);
                res.status(500).send(error.message);
            }
            else if (stderr) {
                console.log(`stderr: ${stderr}`);
                res.status(500).send(stderr);
            }
            else {
                console.log(`stdout: ${stdout}`);
                res.status(200).send(stdout);
            }
        });
        tesseract.on('close', (code) => {
            console.log(`Tesseract exited with code ${code}`);
            // Delete file from local storage
            fs.rm(filePath, () => { console.log(`${fileName} deleted from local storage`) });
        });
    });

});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});