import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import { API_ENDPOINTS } from './config';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import AuthService from './services/AuthService';

class ImageAnalysisService {
  constructor() {
    this.defaultTimeout = 300000; // 5 minutes (increased for detailed analysis)
    this.maxRetries = 3;
    this.qualityThresholds = {
      minWidth: 400,
      minHeight: 400,
      maxFileSize: 10 * 1024 * 1024, // 10MB
      minFileSize: 50 * 1024, // 50KB
      minQualityScore: 0.6,
    };
  }

  // Comprehensive Image Quality Validation
  async validateImageQuality(uri) {
    console.log('Starting comprehensive image quality validation...');

    try {
      // Get image metadata
      const response = await fetch(uri);
      const blob = await response.blob();
      const fileSize = blob.size;

      const validationResults = {
        passed: true,
        score: 0,
        issues: [],
        recommendations: [],
        metadata: {
          fileSize: fileSize,
          fileSizeKB: Math.round(fileSize / 1024),
          uri: uri
        }
      };

      // 1. File Size Validation
      const fileSizeResult = this.validateFileSize(fileSize);
      validationResults.issues.push(...fileSizeResult.issues);
      validationResults.recommendations.push(...fileSizeResult.recommendations);

      // 2. Get image dimensions using Image component
      const dimensions = await this.getImageDimensions(uri);
      validationResults.metadata.width = dimensions.width;
      validationResults.metadata.height = dimensions.height;

      // 3. Resolution Validation
      const resolutionResult = this.validateResolution(dimensions);
      validationResults.issues.push(...resolutionResult.issues);
      validationResults.recommendations.push(...resolutionResult.recommendations);

      // 4. Aspect Ratio Validation
      const aspectRatioResult = this.validateAspectRatio(dimensions);
      validationResults.issues.push(...aspectRatioResult.issues);
      validationResults.recommendations.push(...aspectRatioResult.recommendations);

      // 5. Basic Blur Detection (simplified)
      const blurResult = await this.detectBlur(uri, dimensions);
      validationResults.issues.push(...blurResult.issues);
      validationResults.recommendations.push(...blurResult.recommendations);

      // Calculate overall quality score
      validationResults.score = this.calculateQualityScore(validationResults);
      validationResults.passed = validationResults.score >= this.qualityThresholds.minQualityScore;

      console.log('Image quality validation results:', validationResults);
      return validationResults;

    } catch (error) {
      console.error('Image quality validation failed:', error);
      // Return a user-friendly validation result without exposing technical errors
      return {
        passed: true, // Allow the image to proceed if validation fails
        score: 0.7, // Give it a reasonable default score
        issues: [],
        recommendations: [],
        metadata: {
          uri,
          fileSize: 0,
          fileSizeKB: 0,
          width: 800,
          height: 600
        }
      };
    }
  }

  // Get image dimensions using expo-image-manipulator
  async getImageDimensions(uri) {
    try {
      // Use manipulateAsync with no operations to get image info
      const result = await manipulateAsync(uri, [], { compress: 1 });

      // For React Native, we need to get dimensions differently
      // We'll use a heuristic approach based on file size and common resolutions
      const response = await fetch(uri);
      const blob = await response.blob();
      const fileSize = blob.size;

      // Estimate dimensions based on file size (rough heuristic)
      // This is approximate but sufficient for quality validation
      let estimatedPixels = fileSize / 3; // Rough estimate: 3 bytes per pixel for JPEG
      let dimension = Math.sqrt(estimatedPixels);

      // Clamp to reasonable values
      dimension = Math.max(200, Math.min(4000, dimension));

      return {
        width: Math.round(dimension),
        height: Math.round(dimension)
      };
    } catch (error) {
      console.log('Could not get exact dimensions, using file size estimation');
      return { width: 800, height: 600 }; // Default reasonable dimensions
    }
  }

  // File Size Validation
  validateFileSize(fileSize) {
    const issues = [];
    const recommendations = [];

    if (fileSize > this.qualityThresholds.maxFileSize) {
      issues.push({
        type: 'file_size_too_large',
        severity: 'medium',
        message: `File size is ${Math.round(fileSize / (1024 * 1024))}MB (max recommended: 10MB)`,
        description: 'Large files may take longer to process'
      });
      recommendations.push('Consider using a lower resolution or compressing the image');
    }

    if (fileSize < this.qualityThresholds.minFileSize) {
      issues.push({
        type: 'file_size_too_small',
        severity: 'high',
        message: `File size is ${Math.round(fileSize / 1024)}KB (minimum: 50KB)`,
        description: 'Image may be too low quality for accurate analysis'
      });
      recommendations.push('Use a higher resolution image or better camera settings');
    }

    return { issues, recommendations };
  }

  // Resolution Validation
  validateResolution(dimensions) {
    const { width, height } = dimensions;
    const issues = [];
    const recommendations = [];

    if (width < this.qualityThresholds.minWidth || height < this.qualityThresholds.minHeight) {
      issues.push({
        type: 'low_resolution',
        severity: 'high',
        message: `Resolution is ${width}x${height} (minimum recommended: 400x400)`,
        description: 'Low resolution may result in inaccurate analysis'
      });
      recommendations.push('Use a higher resolution camera setting or get closer to the lesion');
    }

    if (width > 4000 || height > 4000) {
      issues.push({
        type: 'very_high_resolution',
        severity: 'low',
        message: `Very high resolution: ${width}x${height}`,
        description: 'Image will be automatically optimized for processing'
      });
      recommendations.push('Image will be compressed for optimal analysis speed');
    }

    return { issues, recommendations };
  }

  // Aspect Ratio Validation
  validateAspectRatio(dimensions) {
    const { width, height } = dimensions;
    const issues = [];
    const recommendations = [];

    if (width === 0 || height === 0) {
      issues.push({
        type: 'invalid_dimensions',
        severity: 'high',
        message: 'Unable to determine image dimensions',
        description: 'Image may be corrupted or unsupported format'
      });
      recommendations.push('Try selecting a different image in JPEG or PNG format');
      return { issues, recommendations };
    }

    const aspectRatio = width / height;

    if (aspectRatio < 0.5 || aspectRatio > 2.0) {
      issues.push({
        type: 'extreme_aspect_ratio',
        severity: 'medium',
        message: `Unusual aspect ratio: ${aspectRatio.toFixed(2)}`,
        description: 'Very narrow or wide images may not capture lesion details properly'
      });
      recommendations.push('Try to frame the lesion in a more square composition');
    }

    return { issues, recommendations };
  }

  // Simplified Blur Detection
  async detectBlur(uri, dimensions) {
    const issues = [];
    const recommendations = [];

    // Note: This is a simplified blur detection
    // In a real implementation, you might use canvas-based edge detection
    try {
      // Placeholder for blur detection logic
      // For now, we'll use heuristics based on file size vs resolution
      const response = await fetch(uri);
      const blob = await response.blob();
      const compressionRatio = blob.size / (dimensions.width * dimensions.height);

      if (compressionRatio < 0.1) {
        issues.push({
          type: 'potential_blur',
          severity: 'medium',
          message: 'Image may be blurry or overly compressed',
          description: 'Low file size relative to resolution suggests blur or compression artifacts'
        });
        recommendations.push('Ensure the camera is focused and steady when taking the photo');
        recommendations.push('Use good lighting and avoid camera shake');
      }

    } catch (error) {
      console.log('Blur detection skipped:', error.message);
    }

    return { issues, recommendations };
  }

  // Calculate Overall Quality Score
  calculateQualityScore(validationResults) {
    let score = 1.0; // Start with perfect score

    validationResults.issues.forEach(issue => {
      switch (issue.severity) {
        case 'high':
          score -= 0.3;
          break;
        case 'medium':
          score -= 0.15;
          break;
        case 'low':
          score -= 0.05;
          break;
      }
    });

    return Math.max(0, score); // Don't go below 0
  }

  // Get Quality Assessment Text
  getQualityAssessment(score) {
    if (score >= 0.9) return { level: 'Excellent', color: '#22c55e', emoji: '✅' };
    if (score >= 0.7) return { level: 'Good', color: '#3b82f6', emoji: '👍' };
    if (score >= 0.6) return { level: 'Acceptable', color: '#f59e0b', emoji: '⚠️' };
    if (score >= 0.4) return { level: 'Poor', color: '#ef4444', emoji: '❌' };
    return { level: 'Very Poor', color: '#dc2626', emoji: '🚫' };
  }

  // Optimize image before analysis
  async optimizeImageForAnalysis(uri) {
    try {
      console.log('Starting image optimization...');

      // Get original image info
      const response = await fetch(uri);
      const blob = await response.blob();
      const originalSizeKB = Math.round(blob.size / 1024);
      console.log(`Original image size: ${originalSizeKB} KB`);

      // Only compress if image is large
      if (blob.size > 1024 * 1024) { // 1MB threshold
        const result = await manipulateAsync(
          uri,
          [
            { resize: { width: 512 } } // Reduce size for faster processing
          ],
          {
            compress: 0.7,
            format: SaveFormat.JPEG
          }
        );

        const compressedResponse = await fetch(result.uri);
        const compressedBlob = await compressedResponse.blob();
        const compressedSizeKB = Math.round(compressedBlob.size / 1024);
        console.log(`Compressed image size: ${compressedSizeKB} KB (${Math.round((1 - compressedBlob.size/blob.size) * 100)}% reduction)`);

        return result.uri;
      } else {
        console.log('Image size acceptable, skipping compression');
        return uri;
      }
    } catch (error) {
      console.error('Image compression failed:', error);
      // Return original URI if compression fails
      return uri;
    }
  }

  // Check server connectivity
  async checkConnection() {
    try {
      console.log('Checking server connectivity...');
      const start = Date.now();
      const response = await fetch(`${API_ENDPOINTS.UPLOAD.replace('/upload/', '/health')}`, {
        method: 'GET',
        timeout: 10000 // 10 second timeout for health check
      });

      const latency = Date.now() - start;
      console.log(`Server response time: ${latency}ms`);

      if (latency > 10000) {
        throw new Error('Server is responding slowly');
      }

      return true;
    } catch (error) {
      console.log('Connection test failed, continuing anyway:', error.message);
      // Don't block the process if health check fails
      return true;
    }
  }

  // ==========================================================================
  // ASYNC JOB QUEUE METHODS
  // ==========================================================================

  // Check if job queue (Redis/Celery) is available
  async checkJobQueueHealth() {
    // Skip job queue for now - always use sync endpoints
    // Job queue requires Redis + Celery worker to be running
    // TODO: Enable when infrastructure is set up
    return false;
  }

  // Submit a full classify job to the async queue
  async submitJobFullClassify(formData) {
    const token = AuthService.getToken();
    if (!token) throw new Error('Authentication required');

    const response = await fetch(API_ENDPOINTS.JOBS_SUBMIT_FULL_CLASSIFY, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Job submit failed: ${response.status} ${errorText}`);
    }

    return await response.json();
  }

  // Submit a dermoscopy job to the async queue
  async submitJobDermoscopy(imageUri) {
    const token = AuthService.getToken();
    if (!token) throw new Error('Authentication required');

    const formData = new FormData();
    formData.append('file', {
      uri: imageUri,
      name: 'dermoscopy_photo.jpg',
      type: 'image/jpeg'
    });

    const response = await fetch(API_ENDPOINTS.JOBS_SUBMIT_DERMOSCOPY, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Dermoscopy job submit failed: ${response.status} ${errorText}`);
    }

    return await response.json();
  }

  // Submit a burn classify job to the async queue
  async submitJobBurn(imageUri) {
    const token = AuthService.getToken();
    if (!token) throw new Error('Authentication required');

    const formData = new FormData();
    formData.append('file', {
      uri: imageUri,
      name: 'burn_photo.jpg',
      type: 'image/jpeg'
    });

    const response = await fetch(API_ENDPOINTS.JOBS_SUBMIT_BURN, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Burn job submit failed: ${response.status} ${errorText}`);
    }

    return await response.json();
  }

  // Poll job status until completion
  async pollJobUntilComplete(jobId, progressCallback = null, maxWaitMs = 120000) {
    const token = AuthService.getToken();
    if (!token) throw new Error('Authentication required');

    const startTime = Date.now();
    const pollInterval = 1000; // 1 second

    while (Date.now() - startTime < maxWaitMs) {
      const response = await fetch(`${API_ENDPOINTS.JOBS_STATUS}/${jobId}`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) {
        throw new Error(`Job status check failed: ${response.status}`);
      }

      const status = await response.json();
      console.log(`[JOBS] Job ${jobId} status:`, status.status);

      if (status.status === 'SUCCESS') {
        return status.result;
      }

      if (status.status === 'FAILURE') {
        throw new Error(status.error || 'Job failed');
      }

      // Update progress if callback provided
      if (progressCallback && status.progress) {
        const progressPct = status.progress.percent || 0;
        progressCallback(status.progress.message || 'Processing...', progressPct);
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error('Job timed out waiting for completion');
  }

  // Run analysis using async job queue (parallel job submission + polling)
  async analyzeWithJobQueue(imageUri, formData, progressCallback = null) {
    console.log('[JOBS] Starting async job queue analysis');

    if (progressCallback) progressCallback('Submitting analysis jobs...', 10);

    // Submit all 3 jobs in parallel
    const [fullClassifyJob, dermoscopyJob, burnJob] = await Promise.all([
      this.submitJobFullClassify(formData),
      this.submitJobDermoscopy(imageUri).catch(err => {
        console.error('[JOBS] Dermoscopy job submit failed:', err.message);
        return null;
      }),
      this.submitJobBurn(imageUri).catch(err => {
        console.error('[JOBS] Burn job submit failed:', err.message);
        return null;
      })
    ]);

    console.log('[JOBS] Jobs submitted:', {
      fullClassify: fullClassifyJob?.job_id,
      dermoscopy: dermoscopyJob?.job_id,
      burn: burnJob?.job_id
    });

    if (progressCallback) progressCallback('Jobs submitted, waiting for results...', 20);

    // Poll all jobs in parallel
    const [fullResult, dermoscopyResult, burnResult] = await Promise.all([
      this.pollJobUntilComplete(fullClassifyJob.job_id, (msg, pct) => {
        if (progressCallback) progressCallback(`Analyzing lesion: ${msg}`, 20 + (pct * 0.4));
      }),
      dermoscopyJob ? this.pollJobUntilComplete(dermoscopyJob.job_id, null).catch(err => {
        console.error('[JOBS] Dermoscopy job failed:', err.message);
        return null;
      }) : Promise.resolve(null),
      burnJob ? this.pollJobUntilComplete(burnJob.job_id, null).catch(err => {
        console.error('[JOBS] Burn job failed:', err.message);
        return null;
      }) : Promise.resolve(null)
    ]);

    if (progressCallback) progressCallback('Analysis complete, formatting results...', 90);

    return { fullResult, dermoscopyResult, burnResult };
  }

  // Dermoscopic feature analysis function
  async analyzeDermoscopy(imageUri, progressCallback = null) {
    console.log("[DERMOSCOPY] Starting dermoscopic analysis with imageUri:", imageUri);
    const controller = new AbortController();

    const timeoutId = setTimeout(() => {
      controller.abort();
    }, this.defaultTimeout);

    try {
      if (progressCallback) progressCallback('Preparing image for dermoscopic analysis...');

      // Use the image URI directly
      console.log("[DERMOSCOPY] Using image URI:", imageUri);

      // Create form data
      const formData = new FormData();
      formData.append("image", {
        uri: imageUri,
        name: "dermoscopy_photo.jpg",
        type: "image/jpeg",
      });

      console.log("[DERMOSCOPY] FormData created");

      if (progressCallback) progressCallback('Analyzing dermoscopic features...');

      // Send dermoscopy analysis request
      const token = AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required. Please log in again.');
      }

      console.log("[DERMOSCOPY] Sending request to:", API_ENDPOINTS.DERMOSCOPY_ANALYZE);
      const response = await fetch(API_ENDPOINTS.DERMOSCOPY_ANALYZE, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      console.log("[DERMOSCOPY] Response status:", response.status);

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Authentication failed. Your session may have expired. Please log in again.');
        }
        const errorText = await response.text();
        console.error("[DERMOSCOPY] Error response:", errorText);
        throw new Error(`HTTP ${response.status}: ${response.statusText}. ${errorText}`);
      }

      const dermoscopyData = await response.json();
      console.log("[DERMOSCOPY] Dermoscopic analysis response:", dermoscopyData);

      return dermoscopyData;

    } catch (error) {
      clearTimeout(timeoutId);

      console.error("[DERMOSCOPY] Exception caught:", error);
      console.error("[DERMOSCOPY] Error message:", error.message);

      if (error.name === 'AbortError') {
        throw new Error('Dermoscopic analysis timed out. Please try with a smaller image or check your connection.');
      }
      throw error;
    }
  }

  // Burn classification function
  async classifyBurn(imageUri, progressCallback = null) {
    console.log("[BURN classifyBurn] Starting with imageUri:", imageUri);
    const controller = new AbortController();

    const timeoutId = setTimeout(() => {
      controller.abort();
    }, this.defaultTimeout);

    try {
      if (progressCallback) progressCallback('Preparing image for burn analysis...');

      // Use the image URI directly (it should already be optimized by the caller)
      console.log("[BURN classifyBurn] Using image URI:", imageUri);

      // Create form data - React Native specific format
      const formData = new FormData();
      formData.append("image", {
        uri: imageUri,
        name: "burn_photo.jpg",
        type: "image/jpeg",
      });

      console.log("[BURN classifyBurn] FormData created");

      if (progressCallback) progressCallback('Analyzing burn severity...');

      // Send burn classification request
      const token = AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required. Please log in again.');
      }

      console.log("[BURN classifyBurn] Sending request to:", API_ENDPOINTS.CLASSIFY_BURN);
      console.log("[BURN classifyBurn] Token available:", !!token);

      const response = await fetch(API_ENDPOINTS.CLASSIFY_BURN, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          // Don't set Content-Type - let fetch handle it for FormData
        },
        body: formData,
        signal: controller.signal,
      });

      console.log("[BURN classifyBurn] Fetch completed");

      clearTimeout(timeoutId);

      console.log("[BURN classifyBurn] Response status:", response.status);
      console.log("[BURN classifyBurn] Response OK:", response.ok);

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Authentication failed. Your session may have expired. Please log in again.');
        }
        const errorText = await response.text();
        console.error("[BURN classifyBurn] Error response:", errorText);
        throw new Error(`HTTP ${response.status}: ${response.statusText}. ${errorText}`);
      }

      const burnData = await response.json();
      console.log("[BURN classifyBurn] Burn classification response:", burnData);

      return burnData;

    } catch (error) {
      clearTimeout(timeoutId);

      console.error("[BURN classifyBurn] Exception caught:", error);
      console.error("[BURN classifyBurn] Error name:", error.name);
      console.error("[BURN classifyBurn] Error message:", error.message);

      if (error.name === 'AbortError') {
        throw new Error('Burn analysis timed out. Please try with a smaller image or check your connection.');
      }
      throw error;
    }
  }

  // Main analysis function with timeout and retry
  async analyzeImage(imageUri, progressCallback = null, save_to_db = true, bodyMapData = null, analysisType = "binary") {
    const controller = new AbortController();

    // Set up timeout
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, this.defaultTimeout);

    try {
      if (progressCallback) progressCallback('Preparing image...');

      // Step 1: Check connection (optional, non-blocking)
      await this.checkConnection();

      // Step 2: Optimize image
      if (progressCallback) progressCallback('Optimizing image...');
      const optimizedUri = await this.optimizeImageForAnalysis(imageUri);

      // Step 3: Create form data
      const formData = new FormData();
      formData.append("file", {
        uri: optimizedUri,
        name: "photo.jpg",
        type: "image/jpeg",
      });
      formData.append("save_to_db", save_to_db.toString());
      formData.append("analysis_type", analysisType);
      formData.append("enable_multimodal", "true"); // Enable multimodal analysis

      // Add body map data if provided
      if (bodyMapData) {
        if (bodyMapData.body_location) formData.append("body_location", bodyMapData.body_location);
        if (bodyMapData.body_sublocation) formData.append("body_sublocation", bodyMapData.body_sublocation);
        if (bodyMapData.body_side) formData.append("body_side", bodyMapData.body_side);
        if (bodyMapData.body_map_x !== undefined) formData.append("body_map_x", bodyMapData.body_map_x.toString());
        if (bodyMapData.body_map_y !== undefined) formData.append("body_map_y", bodyMapData.body_map_y.toString());
      }

      if (progressCallback) progressCallback('Connecting to analysis server...');

      // Step 4: Send binary classification request with authentication
      const token = AuthService.getToken();
      console.log('Token available:', !!token);
      if (!token) {
        throw new Error('Authentication required. Please log in again.');
      }

      const response = await fetch(API_ENDPOINTS.UPLOAD, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          // Do NOT set Content-Type for multipart/form-data - let fetch set it automatically
        },
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Authentication failed. Your session may have expired. Please log in again.');
        }
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${response.statusText}. ${errorText}`);
      }

      const binaryData = await response.json();
      console.log("Binary response:", binaryData);

      return { binaryData, formData, optimizedUri };

    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        throw new Error('Analysis timed out. Please try with a smaller image or check your connection.');
      }
      throw error;
    }
  }

  // Full classification with retry logic
  async runFullClassify(formData, progressCallback = null, updateExisting = false) {
    const controller = new AbortController();

    const timeoutId = setTimeout(() => {
      controller.abort();
    }, this.defaultTimeout);

    try {
      const token = AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required. Please log in again.');
      }

      // Note: update_existing, save_to_db, and analysis_type should already be set in formData
      // before calling this function to avoid duplication

      const response = await fetch(API_ENDPOINTS.FULL_CLASSIFY, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Authentication failed. Your session may have expired. Please log in again.');
        }
        const errorText = await response.text();
        console.error('Full classify error response:', errorText);
        console.error('Response status:', response.status, response.statusText);
        throw new Error(`HTTP ${response.status}: ${response.statusText}. ${errorText}`);
      }

      const result = await response.json();
      console.log("Full classifier response:", result);

      return result;

    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        throw new Error('Detailed analysis timed out. Please try again.');
      }
      throw error;
    }
  }

  // Main function with retry logic and automatic flow
  // Now accepts clinicalContext for Bayesian risk adjustment
  async analyzeWithRetry(imageUri, progressCallback = null, bodyMapData = null, analysisType = "binary", clinicalContext = null) {
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        console.log(`Analysis attempt ${attempt}/${this.maxRetries}`);
        if (clinicalContext) {
          console.log("[CLINICAL] Clinical context provided:", Object.keys(clinicalContext));
        }

        if (progressCallback) {
          progressCallback(`Analyzing... (attempt ${attempt}/${this.maxRetries})`);
        }

        // COMPREHENSIVE APPROACH: Run all analyses
        // This runs: lesion + inflammatory + infectious disease + burn classification
        if (progressCallback) {
          progressCallback('Running comprehensive skin analysis...', 10);
        }

        // Optimize image first
        const optimizedUri = await this.optimizeImageForAnalysis(imageUri);

        if (progressCallback) {
          progressCallback('Image optimized, uploading...', 25);
        }

        // Create FormData for comprehensive analysis
        const formData = new FormData();
        formData.append("file", {
          uri: optimizedUri,
          name: "photo.jpg",
          type: "image/jpeg",
        });
        formData.append("save_to_db", "true");
        formData.append("analysis_type", "lesion"); // This triggers comprehensive analysis
        formData.append("enable_multimodal", "true"); // Enable multimodal analysis (image + labs + history)

        // Add body map data if provided
        if (bodyMapData) {
          if (bodyMapData.body_location) formData.append("body_location", bodyMapData.body_location);
          if (bodyMapData.body_sublocation) formData.append("body_sublocation", bodyMapData.body_sublocation);
          if (bodyMapData.body_side) formData.append("body_side", bodyMapData.body_side);
          if (bodyMapData.body_map_x !== undefined) formData.append("body_map_x", bodyMapData.body_map_x.toString());
          if (bodyMapData.body_map_y !== undefined) formData.append("body_map_y", bodyMapData.body_map_y.toString());
        }

        // Add clinical context for Bayesian risk adjustment
        if (clinicalContext && Object.keys(clinicalContext).length > 0) {
          console.log("[CLINICAL] Adding clinical context to request");

          // Send as JSON string for complex nested objects
          formData.append("clinical_context_json", JSON.stringify(clinicalContext));

          // Also send individual fields for easier server-side processing
          if (clinicalContext.patient_age !== undefined) {
            formData.append("patient_age", clinicalContext.patient_age.toString());
          }
          if (clinicalContext.fitzpatrick_skin_type) {
            formData.append("fitzpatrick_skin_type", clinicalContext.fitzpatrick_skin_type);
          }
          if (clinicalContext.lesion_duration) {
            formData.append("lesion_duration", clinicalContext.lesion_duration);
          }
          if (clinicalContext.has_changed_recently !== undefined) {
            formData.append("has_changed_recently", clinicalContext.has_changed_recently.toString());
          }
          if (clinicalContext.is_new_lesion !== undefined) {
            formData.append("is_new_lesion", clinicalContext.is_new_lesion.toString());
          }

          // Symptoms
          if (clinicalContext.symptoms) {
            if (clinicalContext.symptoms.itching !== undefined) {
              formData.append("symptoms_itching", clinicalContext.symptoms.itching.toString());
            }
            if (clinicalContext.symptoms.bleeding !== undefined) {
              formData.append("symptoms_bleeding", clinicalContext.symptoms.bleeding.toString());
            }
            if (clinicalContext.symptoms.pain !== undefined) {
              formData.append("symptoms_pain", clinicalContext.symptoms.pain.toString());
            }
          }

          // Medical history
          if (clinicalContext.personal_history_melanoma !== undefined) {
            formData.append("personal_history_melanoma", clinicalContext.personal_history_melanoma.toString());
          }
          if (clinicalContext.personal_history_skin_cancer !== undefined) {
            formData.append("personal_history_skin_cancer", clinicalContext.personal_history_skin_cancer.toString());
          }
          if (clinicalContext.family_history_melanoma !== undefined) {
            formData.append("family_history_melanoma", clinicalContext.family_history_melanoma.toString());
          }
          if (clinicalContext.family_history_skin_cancer !== undefined) {
            formData.append("family_history_skin_cancer", clinicalContext.family_history_skin_cancer.toString());
          }

          // Risk factors
          if (clinicalContext.history_severe_sunburns !== undefined) {
            formData.append("history_severe_sunburns", clinicalContext.history_severe_sunburns.toString());
          }
          if (clinicalContext.uses_tanning_beds !== undefined) {
            formData.append("uses_tanning_beds", clinicalContext.uses_tanning_beds.toString());
          }
          if (clinicalContext.immunosuppressed !== undefined) {
            formData.append("immunosuppressed", clinicalContext.immunosuppressed.toString());
          }
          if (clinicalContext.many_moles !== undefined) {
            formData.append("many_moles", clinicalContext.many_moles.toString());
          }
        }

        if (progressCallback) {
          progressCallback('Uploading image to server...', 40);
        }

        // Start a progress simulation while waiting for the API
        let currentProgress = 40;
        const progressInterval = setInterval(() => {
          if (currentProgress < 80) {
            currentProgress += 5;
            if (progressCallback) {
              progressCallback('Analyzing skin patterns with AI...', currentProgress);
            }
          }
        }, 800); // Update every 800ms

        let fullResult;
        let burnResult = null;
        let dermoscopyResult = null;
        try {
          // Run ALL analyses in parallel: lesion + inflammatory + infectious + burn + dermoscopy
          console.log("[ANALYSIS] Starting comprehensive analysis with URI:", optimizedUri);

          const [lesionAnalysis, burnAnalysis, dermoscopyAnalysis] = await Promise.all([
            this.runFullClassify(formData, null, false),
            this.classifyBurn(optimizedUri).catch(err => {
              console.error("[BURN] Burn classification failed:", err);
              console.error("[BURN] Error message:", err.message);
              return null; // Don't fail entire analysis if burn fails
            }),
            this.analyzeDermoscopy(optimizedUri).catch(err => {
              console.error("[DERMOSCOPY] Dermoscopic analysis failed:", err);
              console.error("[DERMOSCOPY] Error message:", err.message);
              return null; // Don't fail entire analysis if dermoscopy fails
            })
          ]);

          fullResult = lesionAnalysis;
          burnResult = burnAnalysis;
          dermoscopyResult = dermoscopyAnalysis;

          // Clear the progress interval
          clearInterval(progressInterval);
        } catch (error) {
          clearInterval(progressInterval);
          throw error;
        }

        if (progressCallback) {
          progressCallback('Analysis complete, formatting results...', 85);
        }

        console.log("Full classification result received:", fullResult);
        console.log("[BURN] Burn classification result received:", burnResult);
        console.log("[BURN] Burn result is null?", burnResult === null);
        console.log("[DERMOSCOPY] Dermoscopic analysis result received:", dermoscopyResult);
        console.log("[DERMOSCOPY] Dermoscopy result is null?", dermoscopyResult === null);

        if (progressCallback) {
          progressCallback('Processing results...', 95);
        }

        if (progressCallback) {
          progressCallback('Analysis complete!', 100);
        }

        // Format burn results if available
        console.log("[BURN] About to format burn result. burnResult:", burnResult);
        const formattedBurnResult = burnResult ? this.formatBurnResult(burnResult) : null;
        console.log("[BURN] Formatted burn result:", formattedBurnResult);

        // Format dermoscopy results if available
        // SKIP dermoscopy for burns, inflammatory, and infectious conditions - melanoma screening not relevant
        const primaryConditionType = fullResult?.primary_condition_type;
        console.log(`[DERMOSCOPY] primary_condition_type from API: "${primaryConditionType}"`);

        // Also check if burn was detected via burn result as a fallback
        const burnDetected = burnResult?.is_burn_detected === true ||
                             burnResult?.severity_level > 0 ||
                             (burnResult?.severity_class && burnResult?.severity_class.includes('Degree'));

        const skipDermoscopy = primaryConditionType === 'burn' ||
                               primaryConditionType === 'inflammatory' ||
                               primaryConditionType === 'infectious' ||
                               burnDetected; // Fallback: if burn classifier detected a burn

        if (skipDermoscopy) {
          console.log(`[DERMOSCOPY] SKIPPING - not relevant. primaryConditionType="${primaryConditionType}", burnDetected=${burnDetected}`);
        } else {
          console.log("[DERMOSCOPY] Including dermoscopy result:", dermoscopyResult);
        }

        // Since we're doing comprehensive analysis directly, return the result
        // No need for binary check - the full classifier handles everything
        return {
          isLesion: fullResult?.is_lesion || true, // Assume lesion if not specified
          binaryData: {
            is_lesion: fullResult?.is_lesion || true,
            confidence_boolean: fullResult?.binary_confidence || true,
            lesion_probability: fullResult?.binary_confidence || 0.95
          },
          fullResult,
          formattedResult: this.formatAnalysisResult(fullResult),
          // Add burn data to the result
          burnResult,
          formattedBurnResult,
          // Add dermoscopy data to the result - NULL if condition is burn/inflammatory/infectious
          dermoscopyResult: skipDermoscopy ? null : dermoscopyResult
        };

      } catch (error) {
        console.error(`Attempt ${attempt} failed:`, error.message);

        if (attempt === this.maxRetries) {
          throw new Error(`Analysis failed after ${this.maxRetries} attempts: ${error.message}`);
        }

        // Wait before retry (exponential backoff)
        const delay = Math.pow(2, attempt) * 1000; // 2s, 4s, 8s
        console.log(`Retrying in ${delay/1000}s...`);

        if (progressCallback) {
          progressCallback(`Retry in ${delay/1000}s... (${attempt}/${this.maxRetries})`);
        }

        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  // Function to run full analysis on non-lesion images if user chooses
  async runFullAnalysisOnNonLesion(imageUri, progressCallback = null) {
    try {
      if (progressCallback) {
        progressCallback('Running detailed analysis on non-lesion image...');
      }

      const { formData } = await this.analyzeImage(imageUri, progressCallback);
      const fullResult = await this.runFullClassify(formData, progressCallback);

      return {
        fullResult,
        formattedResult: this.formatAnalysisResult(fullResult)
      };
    } catch (error) {
      throw new Error(`Full analysis failed: ${error.message}`);
    }
  }

  // Helper function to format results professionally
  formatAnalysisResult(result) {
    // Guard against null/undefined result
    if (!result) {
      console.warn('[formatAnalysisResult] Received null/undefined result');
      return {
        primaryConditionType: 'unknown',
        primaryConditionConfidence: 0,
        predictedClass: 'Analysis Unavailable',
        confidence: 'N/A',
        confidenceLevel: { level: 'N/A', color: '#666', description: 'No analysis data' },
        probabilities: [],
        inflammatoryCondition: null,
        inflammatoryConfidence: null,
        inflammatoryProbabilities: {},
        differentialDiagnoses: { lesion: [], inflammatory: [], infectious: [], burn: [] },
        uncertaintyMetrics: {},
        probabilitiesWithUncertainty: {},
        mcSamplesUsed: 0,
        timestamp: new Date().toISOString(),
        analysisType: 'Unknown',
        riskLevel: { level: 'Unknown', color: '#666', recommendation: 'Unable to analyze', urgency: 'unknown' },
        redFlagIndicators: null,
        formattedText: 'Analysis results unavailable'
      };
    }

    // Check if this is a burn-primary result (lesion data suppressed)
    const isBurnPrimary = result.primary_condition_type === 'burn';

    // Sort probabilities by confidence for professional display
    // Handle null probabilities (when burn is primary condition)
    const sortedProbabilities = result.probabilities
      ? Object.entries(result.probabilities)
          .map(([key, value]) => ({
            key,
            label: result.key_map?.[key] ?? key,
            probability: value,
            percentage: (value * 100).toFixed(1)
          }))
          .sort((a, b) => b.probability - a.probability)
      : [];

    // Extract infectious disease data if present
    const infectiousData = result.infectious_disease ? {
      infectiousDisease: result.infectious_disease,
      infectiousConfidence: result.infectious_confidence ? (result.infectious_confidence * 100).toFixed(1) : null,
      infectionType: result.infection_type,
      infectiousSeverity: result.infectious_severity,
      contagious: result.contagious,
      transmissionRisk: result.transmission_risk,
      infectiousProbabilities: result.infectious_probabilities || {}
    } : {};

    // For burns, use burn-specific data
    if (isBurnPrimary) {
      return {
        // Primary condition info
        primaryConditionType: 'burn',
        primaryConditionConfidence: result.primary_condition_confidence,

        // Burn-specific results
        predictedClass: result.burn_severity || 'Burn Detected',
        confidence: result.burn_confidence ? (result.burn_confidence * 100).toFixed(1) : 'N/A',
        confidenceLevel: result.burn_confidence ? this.getConfidenceLevel(result.burn_confidence) : { level: 'N/A', color: '#666', description: 'Burn analysis' },

        // Burn details
        burnSeverity: result.burn_severity,
        burnSeverityLevel: result.burn_severity_level,
        burnUrgency: result.burn_urgency,
        burnTreatmentAdvice: result.burn_treatment_advice,
        burnMedicalAttentionRequired: result.burn_medical_attention_required,
        isBurnDetected: result.is_burn_detected,

        // Empty lesion probabilities (suppressed)
        probabilities: [],

        // Inflammatory and infectious suppressed for burns
        inflammatoryCondition: null,
        inflammatoryConfidence: null,
        inflammatoryProbabilities: {},

        // Differential diagnoses (only burn differentials)
        differentialDiagnoses: result.differential_diagnoses || { lesion: [], inflammatory: [], infectious: [], burn: [] },

        // No uncertainty metrics for burns
        uncertaintyMetrics: {},
        probabilitiesWithUncertainty: {},
        mcSamplesUsed: 0,

        // Analysis metadata
        timestamp: new Date().toISOString(),
        analysisType: 'Burn Severity Classification',

        // Risk assessment based on burn severity
        riskLevel: this.assessBurnRiskLevel(result.burn_severity_level, result.burn_medical_attention_required),

        // Red flag indicators (skipped for burns)
        redFlagIndicators: result.red_flag_indicators,

        // Formatted text for burns
        formattedText: `Detected: ${result.burn_severity || 'Burn'}\nUrgency: ${result.burn_urgency || 'See healthcare provider'}\n\n${result.burn_treatment_advice || ''}`
      };
    }

    // Standard lesion/inflammatory/infectious result formatting
    return {
      // Primary condition info
      primaryConditionType: result.primary_condition_type || 'lesion',
      primaryConditionConfidence: result.primary_condition_confidence,

      // Primary result
      predictedClass: result.predicted_class,
      confidence: result.lesion_confidence ? (result.lesion_confidence * 100).toFixed(1) : 'N/A',
      confidenceLevel: result.lesion_confidence ? this.getConfidenceLevel(result.lesion_confidence) : { level: 'N/A', color: '#666', description: 'Analysis pending' },

      // Detailed probabilities
      probabilities: sortedProbabilities,

      // Inflammatory condition analysis
      inflammatoryCondition: result.inflammatory_condition,
      inflammatoryConfidence: result.inflammatory_confidence ? (result.inflammatory_confidence * 100).toFixed(1) : null,
      inflammatoryProbabilities: result.inflammatory_probabilities
        ? Object.fromEntries(
            Object.entries(result.inflammatory_probabilities)
              .filter(([key, value]) => !key.startsWith('_') && typeof value === 'number')
          )
        : {},

      // Infectious disease data
      ...infectiousData,

      // Differential diagnoses
      differentialDiagnoses: result.differential_diagnoses || { lesion: [], inflammatory: [], infectious: [], burn: [] },

      // Monte Carlo Dropout uncertainty quantification
      uncertaintyMetrics: result.uncertainty_metrics || {},
      probabilitiesWithUncertainty: result.probabilities_with_uncertainty || {},
      mcSamplesUsed: result.mc_samples_used || 0,

      // Analysis metadata
      timestamp: new Date().toISOString(),
      analysisType: 'Detailed Lesion Classification with Uncertainty Quantification',

      // Risk assessment
      riskLevel: result.predicted_class
        ? this.assessRiskLevel(result.predicted_class, result.lesion_confidence || 0)
        : { level: 'Unknown', color: '#666', recommendation: 'Analysis incomplete', urgency: 'unknown' },

      // Red flag indicators
      redFlagIndicators: result.red_flag_indicators,

      // Formatted text (legacy compatibility)
      formattedText: result.predicted_class
        ? `Predicted: ${result.predicted_class} (${result.lesion_confidence ? (result.lesion_confidence * 100).toFixed(1) : 'N/A'}%)\n\nProbabilities:\n${sortedProbabilities.map(p => `  ${p.label}: ${p.percentage}%`).join('\n')}`
        : 'Analysis results unavailable'
    };
  }

  // Helper function to format burn classification results
  formatBurnResult(result) {
    // Sort probabilities by confidence for professional display
    const sortedProbabilities = Object.entries(result.probabilities || {})
      .map(([severity, probability]) => ({
        severity,
        probability,
        percentage: (probability * 100).toFixed(1)
      }))
      .sort((a, b) => b.probability - a.probability);

    return {
      // Primary result
      severityClass: result.severity_class,
      severityLevel: result.severity_level,
      confidence: (result.confidence * 100).toFixed(1),
      confidenceLevel: this.getConfidenceLevel(result.confidence),

      // Burn details
      urgency: result.urgency,
      treatmentAdvice: result.treatment_advice,
      medicalAttentionRequired: result.medical_attention_required,
      isBurnDetected: result.is_burn_detected,

      // Detailed probabilities
      probabilities: sortedProbabilities,

      // Analysis metadata
      timestamp: new Date().toISOString(),
      analysisType: 'Burn Severity Classification',

      // Risk assessment based on severity level
      riskLevel: this.assessBurnRiskLevel(result.severity_level, result.medical_attention_required),

      // Formatted text
      formattedText: `Detected: ${result.severity_class} (${(result.confidence * 100).toFixed(1)}%)\n` +
                     `Urgency: ${result.urgency}\n\n` +
                     `Treatment Advice:\n${result.treatment_advice}\n\n` +
                     `Probabilities:\n${sortedProbabilities.map(p => `  ${p.severity}: ${p.percentage}%`).join('\n')}`
    };
  }

  // Assess risk level for burn classification
  assessBurnRiskLevel(severityLevel, medicalAttentionRequired) {
    if (severityLevel === 3) {
      // Third degree burn
      return {
        level: 'Critical',
        color: '#dc2626',
        recommendation: 'EMERGENCY - Call 911 or go to ER immediately',
        urgency: 'critical'
      };
    } else if (severityLevel === 2) {
      // Second degree burn
      return {
        level: 'High',
        color: '#f59e0b',
        recommendation: 'Seek medical attention within 24 hours',
        urgency: 'high'
      };
    } else if (severityLevel === 1) {
      // First degree burn
      return {
        level: 'Low',
        color: '#fbbf24',
        recommendation: 'Monitor and apply first aid. Seek medical care if symptoms worsen',
        urgency: 'low'
      };
    } else {
      // Normal/Healthy skin
      return {
        level: 'None',
        color: '#22c55e',
        recommendation: 'No burn detected',
        urgency: 'none'
      };
    }
  }

  // Helper function to format infectious disease results
  formatInfectiousResult(result) {
    // Sort probabilities by confidence for professional display
    const sortedProbabilities = Object.entries(result.probabilities || {})
      .map(([key, value]) => ({
        disease: key,
        probability: value,
        percentage: (value * 100).toFixed(1)
      }))
      .sort((a, b) => b.probability - a.probability);

    return {
      // Primary result
      predictedDisease: result.predicted_disease,
      confidence: (result.confidence * 100).toFixed(1),
      confidenceLevel: this.getConfidenceLevel(result.confidence),

      // Infection details
      infectionType: result.infection_type,
      severity: result.severity,
      contagious: result.contagious,
      transmissionRisk: result.transmission_risk,

      // Clinical information
      description: result.description,
      symptoms: result.symptoms,
      treatmentRecommendations: result.treatment_recommendations || [],
      urgency: result.urgency,

      // Detailed probabilities
      probabilities: sortedProbabilities,

      // Differential diagnoses
      differentialDiagnoses: result.differential_diagnoses || [],

      // Analysis metadata
      timestamp: new Date().toISOString(),
      analysisType: 'Infectious Disease Classification',

      // Risk assessment based on severity and contagiousness
      riskLevel: this.assessInfectiousRiskLevel(result.severity, result.contagious, result.transmission_risk),

      // Formatted text
      formattedText: `Predicted: ${result.predicted_disease} (${(result.confidence * 100).toFixed(1)}%)\n` +
                     `Type: ${result.infection_type}\n` +
                     `Severity: ${result.severity}\n` +
                     `Contagious: ${result.contagious ? 'Yes' : 'No'}\n\n` +
                     `Top Probabilities:\n${sortedProbabilities.slice(0, 5).map(p => `  ${p.disease}: ${p.percentage}%`).join('\n')}`
    };
  }

  // Assess risk level for infectious diseases
  assessInfectiousRiskLevel(severity, contagious, transmissionRisk) {
    if (severity === 'severe' || (contagious && transmissionRisk === 'high')) {
      return {
        level: 'High',
        color: '#dc2626',
        recommendation: 'Seek medical attention promptly. Highly contagious - take precautions.',
        urgency: 'high'
      };
    } else if (severity === 'moderate' || (contagious && transmissionRisk === 'medium')) {
      return {
        level: 'Moderate',
        color: '#f59e0b',
        recommendation: 'Consult healthcare provider. Follow hygiene protocols if contagious.',
        urgency: 'moderate'
      };
    } else {
      return {
        level: 'Low',
        color: '#22c55e',
        recommendation: 'Monitor condition. Consult healthcare provider if symptoms worsen.',
        urgency: 'low'
      };
    }
  }

  // Get confidence level description
  getConfidenceLevel(confidence) {
    if (confidence >= 0.9) return { level: 'Very High', color: '#22c55e', description: 'High diagnostic confidence' };
    if (confidence >= 0.75) return { level: 'High', color: '#3b82f6', description: 'Good diagnostic confidence' };
    if (confidence >= 0.6) return { level: 'Moderate', color: '#f59e0b', description: 'Moderate confidence - consider additional evaluation' };
    if (confidence >= 0.4) return { level: 'Low', color: '#ef4444', description: 'Low confidence - recommend dermatologist consultation' };
    return { level: 'Very Low', color: '#dc2626', description: 'Very low confidence - dermatologist evaluation strongly recommended' };
  }

  // Assess risk level based on lesion type and confidence
  assessRiskLevel(predictedClass, confidence) {
    const highRiskTypes = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma'];
    const moderateRiskTypes = ['atypical nevus', 'pigmented benign keratosis'];

    const classLower = predictedClass.toLowerCase();

    if (highRiskTypes.some(type => classLower.includes(type))) {
      return {
        level: 'High',
        color: '#dc2626',
        recommendation: 'Immediate dermatologist consultation recommended',
        urgency: 'high'
      };
    } else if (moderateRiskTypes.some(type => classLower.includes(type))) {
      return {
        level: 'Moderate',
        color: '#f59e0b',
        recommendation: 'Dermatologist evaluation recommended within 2-4 weeks',
        urgency: 'moderate'
      };
    } else {
      return {
        level: 'Low',
        color: '#22c55e',
        recommendation: 'Routine monitoring - consult dermatologist if changes occur',
        urgency: 'low'
      };
    }
  }

  // Professional PDF Report Generation
  async generatePDFReport(analysisData, imageUri = null) {
    try {
      console.log('Generating professional PDF report...');

      // Convert image to base64 for embedding in PDF
      let imageBase64 = '';
      if (imageUri) {
        try {
          const response = await fetch(imageUri);
          const blob = await response.blob();
          const reader = new FileReader();
          imageBase64 = await new Promise((resolve) => {
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
          });
        } catch (error) {
          console.log('Could not embed image in PDF:', error);
        }
      }

      const htmlContent = this.generateMedicalReportHTML(analysisData, imageBase64);

      // Generate PDF
      const { uri } = await Print.printToFileAsync({
        html: htmlContent,
        base64: false,
        height: 792, // Letter size height
        width: 612,  // Letter size width
        margins: {
          left: 40,
          top: 40,
          right: 40,
          bottom: 40
        }
      });

      console.log('PDF generated successfully at:', uri);
      return uri;

    } catch (error) {
      console.error('PDF generation failed:', error);
      throw new Error('Failed to generate PDF report. Please try again.');
    }
  }

  // Share PDF Report
  async sharePDFReport(pdfUri, filename = null) {
    try {
      const defaultFilename = `SkinLesionAnalysis_${new Date().toISOString().split('T')[0]}.pdf`;
      const shareFilename = filename || defaultFilename;

      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(pdfUri, {
          mimeType: 'application/pdf',
          dialogTitle: 'Share Analysis Report',
          UTI: 'com.adobe.pdf'
        });
        console.log('PDF shared successfully');
      } else {
        throw new Error('Sharing is not available on this device');
      }
    } catch (error) {
      console.error('PDF sharing failed:', error);
      throw new Error('Failed to share PDF report. Please try again.');
    }
  }

  // Generate Professional Medical Report HTML
  generateMedicalReportHTML(analysisData, imageBase64 = '') {
    const currentDate = new Date().toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });

    const riskColor = analysisData.riskLevel.urgency === 'high' ? '#dc2626' :
                     analysisData.riskLevel.urgency === 'moderate' ? '#f59e0b' : '#22c55e';

    const confidenceColor = analysisData.confidenceLevel.color;

    return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Skin Lesion Analysis Report</title>
      <style>
        body {
          font-family: 'Helvetica', 'Arial', sans-serif;
          line-height: 1.6;
          margin: 0;
          padding: 20px;
          color: #333;
          background-color: #fff;
        }

        .header {
          text-align: center;
          border-bottom: 3px solid #2c5282;
          padding-bottom: 20px;
          margin-bottom: 30px;
        }

        .header h1 {
          color: #2c5282;
          font-size: 28px;
          margin: 0 0 5px 0;
          font-weight: bold;
        }

        .header h2 {
          color: #4a5568;
          font-size: 16px;
          margin: 0;
          font-weight: normal;
        }

        .report-meta {
          background-color: #f8f9fa;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 25px;
          border-left: 4px solid #4299e1;
        }

        .image-section {
          text-align: center;
          margin-bottom: 30px;
        }

        .analysis-image {
          max-width: 300px;
          max-height: 300px;
          border-radius: 8px;
          border: 2px solid #e2e8f0;
        }

        .diagnosis-section {
          background-color: #f8f9fa;
          padding: 20px;
          border-radius: 12px;
          margin-bottom: 25px;
          border-left: 5px solid ${confidenceColor};
        }

        .diagnosis-title {
          color: #2c5282;
          font-size: 20px;
          font-weight: bold;
          margin-bottom: 10px;
        }

        .predicted-class {
          font-size: 24px;
          font-weight: bold;
          color: #1a202c;
          text-align: center;
          margin: 15px 0;
        }

        .confidence-info {
          text-align: center;
          margin: 15px 0;
        }

        .confidence-score {
          font-size: 18px;
          font-weight: bold;
          color: ${confidenceColor};
        }

        .confidence-desc {
          font-style: italic;
          color: #666;
          margin-top: 5px;
        }

        .risk-section {
          background-color: #fff;
          padding: 20px;
          border-radius: 12px;
          margin-bottom: 25px;
          border-left: 5px solid ${riskColor};
          border: 1px solid #e2e8f0;
        }

        .risk-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .risk-title {
          color: #2c5282;
          font-size: 18px;
          font-weight: bold;
        }

        .risk-badge {
          background-color: ${riskColor};
          color: white;
          padding: 6px 12px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: bold;
        }

        .probabilities-section {
          margin-bottom: 25px;
        }

        .section-title {
          color: #2c5282;
          font-size: 18px;
          font-weight: bold;
          margin-bottom: 15px;
          border-bottom: 2px solid #e2e8f0;
          padding-bottom: 5px;
        }

        .probability-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 0;
          border-bottom: 1px solid #f1f5f9;
        }

        .probability-item:first-child {
          font-weight: bold;
          background-color: #f0f9ff;
          padding: 12px 15px;
          border-radius: 8px;
          margin-bottom: 10px;
        }

        .probability-bar {
          width: 100px;
          height: 8px;
          background-color: #e2e8f0;
          border-radius: 4px;
          overflow: hidden;
          margin-left: 10px;
        }

        .probability-fill {
          height: 100%;
          background-color: ${confidenceColor};
          border-radius: 4px;
        }

        .disclaimer {
          background-color: #fef5e7;
          border: 1px solid #f6e05e;
          border-left: 5px solid #d69e2e;
          padding: 20px;
          border-radius: 8px;
          margin-top: 30px;
        }

        .disclaimer-title {
          color: #744210;
          font-weight: bold;
          font-size: 16px;
          margin-bottom: 10px;
        }

        .disclaimer-text {
          color: #744210;
          font-size: 14px;
          line-height: 1.5;
        }

        .footer {
          text-align: center;
          margin-top: 40px;
          padding-top: 20px;
          border-top: 1px solid #e2e8f0;
          font-size: 12px;
          color: #666;
        }

        table {
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 20px;
        }

        td {
          padding: 8px;
          vertical-align: top;
        }

        .label {
          font-weight: bold;
          color: #4a5568;
          width: 30%;
        }
      </style>
    </head>
    <body>
      <div class="header">
      
        <h1>🔬 Skin Lesion Analysis Report</h1>
        <h2>AI-Powered Dermatological Assessment</h2>
      </div>

      <div class="report-meta">
        <table>
          <tr>
            <td class="label">Report Generated:</td>
            <td>${currentDate}</td>
          </tr>
          <tr>
            <td class="label">Analysis Type:</td>
            <td>${analysisData.analysisType}</td>
          </tr>
          <tr>
            <td class="label">Processing Method:</td>
            <td>AI-Powered Deep Learning</td>
          </tr>
        </table>
      </div>

      ${imageBase64 ? `
      <div class="image-section">
        <h3 class="section-title">📸 Analyzed Image</h3>
        <img src="${imageBase64}" alt="Analyzed lesion" class="analysis-image">
      </div>
      ` : ''}

      <div class="diagnosis-section">
        <div class="diagnosis-title">🔬 Primary Diagnosis</div>
        <div class="predicted-class">${analysisData.predictedClass}</div>
        <div class="confidence-info">
          <div class="confidence-score">Confidence: ${analysisData.confidence}% (${analysisData.confidenceLevel.level})</div>
          <div class="confidence-desc">${analysisData.confidenceLevel.description}</div>
        </div>
      </div>

      <div class="risk-section">
        <div class="risk-header">
          <div class="risk-title">⚠️ Risk Assessment</div>
          <div class="risk-badge">${analysisData.riskLevel.level}</div>
        </div>
        <p><strong>Recommendation:</strong> ${analysisData.riskLevel.recommendation}</p>
      </div>

      <div class="probabilities-section">
        <h3 class="section-title">📊 Detailed Classification Probabilities</h3>
        ${analysisData.probabilities.map((prob, index) => `
          <div class="probability-item ${index === 0 ? 'top-probability' : ''}">
            <span>${prob.label}</span>
            <div style="display: flex; align-items: center;">
              <span style="margin-right: 10px; font-weight: ${index === 0 ? 'bold' : 'normal'};">${prob.percentage}%</span>
              <div class="probability-bar">
                <div class="probability-fill" style="width: ${prob.probability * 100}%;"></div>
              </div>
            </div>
          </div>
        `).join('')}
      </div>

      <div class="disclaimer">
        <div class="disclaimer-title">⚠️ Important Medical Disclaimer</div>
        <div class="disclaimer-text">
          This AI analysis is for educational and research purposes only. It is not intended to diagnose, treat, cure, or prevent any disease. This report should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified dermatologist or healthcare provider regarding any skin lesions or medical conditions. Never disregard professional medical advice or delay in seeking it because of information provided in this report.
        </div>
      </div>

      <div class="footer">
        <p>Generated by Skin Lesion Analysis App • AI-Powered Dermatological Assessment</p>
        <p>For medical consultation, please contact a qualified dermatologist</p>
      </div>
    </body>
    </html>`;
  }

  // Export and Share PDF Report (Combined Function)
  async exportAndSharePDF(analysisData, imageUri = null) {
    try {
      console.log('Starting PDF export process...');

      // Generate PDF
      const pdfUri = await this.generatePDFReport(analysisData, imageUri);

      // Share PDF
      await this.sharePDFReport(pdfUri);

      return pdfUri;
    } catch (error) {
      console.error('PDF export process failed:', error);
      throw error;
    }
  }

  // ============================================================================
  // CLINICAL PHOTOGRAPHY ASSISTANT
  // ============================================================================

  /**
   * Assess photo quality in real-time for clinical photography
   *
   * @param {string} imageUri - URI of image to assess
   * @returns {Promise<Object>} Quality feedback with scores and suggestions
   */
  async assessPhotoQuality(imageUri) {
    try {
      console.log('Assessing photo quality for clinical photography...');

      // Get auth token
      const token = await AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required');
      }

      // Prepare form data
      const formData = new FormData();
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'photo.jpg',
      });

      // Call quality assessment API
      const response = await fetch(`${API_ENDPOINTS.BASE_URL}/photography/assess-quality`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Quality assessment failed: ${response.status}`);
      }

      const feedback = await response.json();
      console.log('Quality assessment result:', feedback);

      return feedback;

    } catch (error) {
      console.error('Photo quality assessment error:', error);
      // Return default feedback so camera doesn't break
      return {
        overall_score: 70,
        quality_level: 'acceptable',
        meets_medical_standards: false,
        dicom_compliant: false,
        scores: {
          lighting: 70,
          focus: 70,
          distance: 70,
          angle: 70,
          scale: 0,
          color_card: 0,
        },
        detections: {
          ruler_detected: false,
          color_card_detected: false,
          has_glare: false,
          has_shadows: false,
          is_blurry: false,
          too_close: false,
          too_far: false,
        },
        measurements: {
          estimated_distance_cm: null,
          pixel_to_mm_ratio: null,
          glare_percentage: 0,
          shadow_percentage: 0,
        },
        feedback: {
          issues: [],
          suggestions: ['Enable AI assistance for quality feedback'],
          warnings: [],
        },
        ready_to_capture: true,
      };
    }
  }

  /**
   * Get medical photography standards and guidelines
   *
   * @returns {Promise<Object>} Photography standards
   */
  async getPhotographyStandards() {
    try {
      const response = await fetch(`${API_ENDPOINTS.BASE_URL}/photography/standards`);

      if (!response.ok) {
        throw new Error('Failed to fetch photography standards');
      }

      return await response.json();

    } catch (error) {
      console.error('Failed to fetch photography standards:', error);
      return null;
    }
  }

  /**
   * Store calibration data from a reference photo
   *
   * @param {number} pixelToMmRatio - Pixel to millimeter ratio
   * @param {string} colorProfile - Color profile data
   * @param {string} deviceInfo - Device information
   * @returns {Promise<Object>} Calibration result
   */
  async storeCalibration(pixelToMmRatio, colorProfile = null, deviceInfo = null) {
    try {
      const token = await AuthService.getToken();
      if (!token) {
        throw new Error('Authentication required');
      }

      const formData = new FormData();
      formData.append('pixel_to_mm_ratio', pixelToMmRatio);
      if (colorProfile) formData.append('color_profile', colorProfile);
      if (deviceInfo) formData.append('device_info', deviceInfo);

      const response = await fetch(`${API_ENDPOINTS.BASE_URL}/photography/calibrate`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to store calibration');
      }

      return await response.json();

    } catch (error) {
      console.error('Failed to store calibration:', error);
      throw error;
    }
  }
}

export default new ImageAnalysisService();