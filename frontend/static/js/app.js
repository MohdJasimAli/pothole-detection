/* ============================================================
   SmartRoad AI - Dashboard JavaScript
   Handles image upload, API communication, and result visualization
   ============================================================ */

document.addEventListener("DOMContentLoaded", function() {
    console.log("SmartRoad AI - Dashboard loaded");

    // Split-deployment support (e.g. dashboard on Netlify, API on Render):
    // set window.POTHOLE_API_BASE to the API origin BEFORE this script
    // loads (see index.html). Empty string = same origin (normal case
    // when Flask serves both the dashboard and the API).
    const API_BASE = window.POTHOLE_API_BASE || "";

    const uploadArea = document.getElementById("uploadArea");
    const imageUpload = document.getElementById("imageUpload");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultsSection = document.getElementById("resultsSection");

    let selectedFile = null;

    // ========================================
    // Upload Area Event Handlers
    // ========================================
    uploadArea.addEventListener("click", function() {
        imageUpload.click();
    });

    uploadArea.addEventListener("dragover", function(e) {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", function() {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", function(e) {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            selectedFile = e.dataTransfer.files[0];
            updateUploadArea(selectedFile.name);
            analyzeBtn.style.display = "block";
        }
    });

    imageUpload.addEventListener("change", function(e) {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            updateUploadArea(selectedFile.name);
            analyzeBtn.style.display = "block";
        }
    });

    function updateUploadArea(filename) {
        const textElement = uploadArea.querySelector(".upload-text");
        textElement.textContent = filename;
        textElement.style.color = "#2d3436";
        textElement.style.fontWeight = "500";
    }

    // ========================================
    // Analyze Button Handler
    // ========================================
    analyzeBtn.addEventListener("click", function() {
        if (!selectedFile) {
            showAlert("Please select an image first");
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-pulse"></i> Analyzing...';

        const formData = new FormData();
        formData.append("image", selectedFile);

        // Get GPS coordinates if provided
        const gpsLat = uploadArea.querySelector('input[name="gps_lat"]').value;
        const gpsLng = uploadArea.querySelector('input[name="gps_lng"]').value;
        if (gpsLat && gpsLng) {
            formData.append("gps_lat", gpsLat);
            formData.append("gps_lng", gpsLng);
        }

        fetch(API_BASE + "/api/analyze", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Analysis failed. Please try again.");
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Image';
        })
        .catch(error => {
            console.error("Error:", error);
            showAlert(error.message);
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Image';
        });
    });

    // ========================================
    // Display Results
    // ========================================
    function displayResults(data) {
        resultsSection.style.display = "block";

        // Update severity summary
        const summary = data.report.severity_summary;
        const summaryHtml = `
            <div class="severity-card critical">
                <div class="count">${summary.critical}</div>
                <div class="label">Critical</div>
            </div>
            <div class="severity-card high">
                <div class="count">${summary.high}</div>
                <div class="label">High</div>
            </div>
            <div class="severity-card medium">
                <div class="count">${summary.medium}</div>
                <div class="label">Medium</div>
            </div>
            <div class="severity-card low">
                <div class="count">${summary.low}</div>
                <div class="label">Low</div>
            </div>
        `;
        document.getElementById("severitySummary").innerHTML = summaryHtml;

        // Display annotated image
        const imgHtml = `<img src="data:image/jpeg;base64,${data.annotated_image_base64}" alt="Pothole Detection Results" style="max-width: 100%;">`;
        document.getElementById("imageComparison").innerHTML = imgHtml;

        // Display pothole list
        let potholeHtml = "";
        data.detections.forEach((det, i) => {
            const severityClass = det.severity ? det.severity.toLowerCase() : "low";
            potholeHtml += `
                <div class="pothole-card ${severityClass}">
                    <h4>
                        Pothole #${i + 1}
                        <span class="pothole-badge ${severityClass}">${det.severity || "Unknown"}</span>
                    </h4>
                    <div class="pothole-detail">
                        <div><span class="label">Confidence:</span> <span class="value">${(det.confidence * 100).toFixed(1)}%</span></div>
                        <div><span class="label">Size:</span> <span class="value">${det.size_cm2} cm²</span></div>
                        <div><span class="label">Dimensions:</span> <span class="value">${det.width_cm} × ${det.height_cm} cm</span></div>
                        <div><span class="label">Size Cat:</span> <span class="value">${det.size_category || "Unknown"}</span></div>
                        <div><span class="label">Depth:</span> <span class="value">${det.depth_cm} cm (${det.depth_category || "Unknown"})</span></div>
                        <div><span class="label">Priority:</span> <span class="value">${det.priority || "N/A"}</span></div>
                        <div><span class="label">Response Time:</span> <span class="value">${det.response_time_hours || "N/A"}h</span></div>
                        <div><span class="label">Processing:</span> <span class="value">${data.processing_time_seconds}s</span></div>
                    </div>
                </div>
            `;
        });
        document.getElementById("potholeList").innerHTML = potholeHtml;

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    // ========================================
    // Load Previous Reports
    // ========================================
    function loadReports() {
        const reportsList = document.getElementById("reportsList");
        if (!reportsList) return;

        fetch(API_BASE + "/api/reports")
            .then(response => response.json())
            .then(data => {
                if (data.reports.length === 0) {
                    reportsList.innerHTML = '<p style="text-align: center; color: #636e72;">No reports yet. Upload an image to generate your first report.</p>';
                    return;
                }

                let html = "";
                data.reports.slice(0, 9).forEach(report => {
                    const date = new Date(report.timestamp).toLocaleDateString();
                    const potholeCount = report.total_potholes;
                    const critical = report.severity_summary?.critical || 0;
                    const high = report.severity_summary?.high || 0;
                    html += `
                        <div class="report-card" onclick="viewReport('${report.report_id}')">
                            <h4>Report #${report.report_id}</h4>
                            <div class="date">${date} • ${potholeCount} pothole(s)</div>
                            <div class="stats">
                                <span class="badge critical">${critical} Critical</span>
                                <span class="badge high">${high} High</span>
                            </div>
                        </div>
                    `;
                });
                reportsList.innerHTML = html;
            })
            .catch(error => {
                console.error("Error loading reports:", error);
            });
    }

    function viewReport(reportId) {
        alert(`Viewing report: ${reportId}\n(Report detail view coming soon)`);
    }

    // ========================================
    // Utilities
    // ========================================
    function showAlert(message) {
        const alertDiv = document.createElement("div");
        alertDiv.style.cssText = `
            position: fixed;
            top: 80px;
            right: 24px;
            background: #e74c3c;
            color: #fff;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            font-weight: 500;
        `;
        alertDiv.textContent = message;
        document.body.appendChild(alertDiv);

        setTimeout(() => {
            alertDiv.remove();
        }, 4000);
    }

    // Load reports on page load
    loadReports();

    // Add badge styles
    const style = document.createElement("style");
    style.textContent = `
        .badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: #fff;
        }
        .badge.critical { background: #e74c3c; }
        .badge.high { background: #e67e22; }
    `;
    document.head.appendChild(style);
});
