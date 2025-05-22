import { useState } from "react";

export default function JobSkillForm() {
  const [jobDescription, setJobDescription] = useState("");
  const [pdfFile, setPdfFile] = useState(null);
  const [rankedSkills, setRankedSkills] = useState([]);
  const [loading, setLoading] = useState(false); // Loading state

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === "application/pdf") {
      setPdfFile(file);
    } else {
      alert("Please upload a PDF file only.");
      setPdfFile(null);
    }
  };

  const handleUpload = async () => {
    if (!pdfFile) {
      alert("No file selected");
      return;
    }

    setLoading(true); // Start loading
    const formData = new FormData();
    formData.append("resume", pdfFile);

    try {
      const response = await fetch("http://localhost:5000/upload_resume", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (result.status === "success") {
        console.log(result);
        setRankedSkills(result.ranked_skills);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Failed to upload PDF.");
    } finally {
      setLoading(false); // Stop loading
    }
  };

  const handleSubmitJD = async () => {
    if (!jobDescription.trim()) {
      alert("Please enter a job description.");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ jobDescription }),
      });

      const result = await response.json();
      if (result.status === "success") {
        console.log(result);
        setRankedSkills(result);
      }
    } catch (error) {
      console.error("Submit error:", error);
      alert("Failed to submit job description.");
    }
  };

  return (
    <>
      <div className="form-container">
        <h1 className="form-title">Upload Resume & Enter Job Description</h1>

        <div className="form-group">
          <label>Upload Resume (PDF only)</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            disabled={loading} // Disable input during upload
          />
          <button onClick={handleUpload} disabled={loading}>
            {loading ? (
              <div className="spinner"></div> // Spinner when loading
            ) : (
              "Upload PDF"
            )}
          </button>
        </div>

        <div className="form-group">
          <label>Job Title</label>
          <textarea
            placeholder="Enter the job title here..."
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
          />
          <button onClick={handleSubmitJD}>Submit Job Description</button>
        </div>
      </div>

      {rankedSkills?.ranked_resumes?.length > 0 ? (
  <div className="results">
    <h2>Ranked Resumes</h2>
    <ol>
      {rankedSkills.ranked_resumes.map((resume, index) => (
        <li key={index}>
          <div>
            <h3>{resume.resume}</h3>
            <p>Score: {resume.score.toFixed(2)}</p>

            {resume.file_name && (
              <a
                href={`http://localhost:5000/download_resume/${encodeURIComponent(resume.file_name)}`}
                download
                target="_blank"
                rel="noopener noreferrer"
              >
                📄 Download Resume
              </a>
            )}
          </div>
        </li>
      ))}
    </ol>
  </div>
) : (
  <div>No ranked resumes to display.</div>
)}


    </>
  );
}
