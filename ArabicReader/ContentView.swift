import SwiftUI
import UIKit
import AVFoundation
import PhotosUI
import Vision
import CoreML

// Replace the existing ContentView.swift with this whole file.
// Add ArabicSymbols.mlmodel to the ArabicReader app target as well.
@MainActor
struct ContentView: View {
    @State private var displayedImage: UIImage?
    @State private var showCamera = false
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var isAnalyzing = false
    @State private var resultSymbol = ""
    @State private var resultName = ""
    @State private var statusMessage = "صوّر رمزًا واحدًا أو اختر صورة لاختباره."
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var worker = SymbolRecognitionWorker()

    // Starting values only; tune with independent real camera photos.
    // These values do NOT guarantee rejection of words or English digits.
    private let minimumConfidence: Float = 0.80
    private let minimumMargin: Float = 0.15

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Text("القارئ العربي")
                    .font(.largeTitle.bold())

                Text("حرف مستقل أو رقم واحد من ٠١٢٣٤٥٦٧٨٩")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)

                if let image = displayedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 320)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                } else {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 64))
                        .foregroundStyle(.blue)
                        .padding(.vertical, 40)
                }

                if isAnalyzing {
                    ProgressView("جارٍ تحليل الصورة…")
                } else if !resultSymbol.isEmpty {
                    Text(resultSymbol)
                        .font(.system(size: 80))
                        .accessibilityLabel(resultName)
                    Text(resultName)
                        .font(.title2.bold())
                }

                Text(statusMessage)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)

                Button(action: openCamera) {
                    Label("التقاط صورة", systemImage: "camera")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isAnalyzing)

                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Label("اختيار صورة من المعرض", systemImage: "photo")
                }
                .buttonStyle(.bordered)
                .disabled(isAnalyzing)
            }
            .padding(24)
        }
        .environment(\.layoutDirection, .rightToLeft)
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { photo in
                analyze(photo)
            }
            .ignoresSafeArea()
        }
        .onChange(of: selectedPhoto) { _, item in
            guard let item else { return }
            isAnalyzing = true
            Task { @MainActor in
                do {
                    guard let data = try await item.loadTransferable(type: Data.self),
                          let image = UIImage(data: data) else {
                        isAnalyzing = false
                        reportError("تعذر فتح الصورة المختارة. جرّب صورة أخرى.")
                        return
                    }
                    analyze(image)
                } catch {
                    isAnalyzing = false
                    reportError("تعذر تحميل الصورة: \(error.localizedDescription)")
                }
            }
        }
        .alert("تعذر إكمال العملية", isPresented: $showError) {
            Button("حسنًا", role: .cancel) { }
        } message: {
            Text(errorMessage)
        }
    }

    private func analyze(_ image: UIImage) {
        resultSymbol = ""
        resultName = ""
        statusMessage = ""
        isAnalyzing = true

        // UIImage drawing applies the camera orientation. Fit the whole
        // selected image on white so scaling doesn't stretch the symbol.
        let prepared = prepareImage(image)
        displayedImage = prepared
        guard let data = prepared.jpegData(compressionQuality: 0.95) else {
            isAnalyzing = false
            reportError("تعذر تجهيز الصورة للتحليل.")
            return
        }

        Task { @MainActor in
            do {
                // The actor performs model loading and inference away from
                // the main actor. Only JPEG bytes cross the actor boundary.
                let prediction = try await worker.predict(data)
                if prediction.confidence >= minimumConfidence,
                   prediction.margin >= minimumMargin,
                   let symbol = symbolForLabel(prediction.identifier),
                   let name = spokenNames[symbol] {
                    resultSymbol = symbol
                    resultName = name
                    statusMessage = "تم التعرف على الرمز."
                } else {
                    statusMessage = "لم يتم التعرف"
                }
            } catch {
                statusMessage = "تعذر تحليل الصورة."
                reportError(error.localizedDescription)
            }
            isAnalyzing = false
        }
    }

    private func prepareImage(_ image: UIImage) -> UIImage {
        let side: CGFloat = 360
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(
            size: CGSize(width: side, height: side), format: format)
        return renderer.image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
            let ratio = min(side / max(image.size.width, 1),
                            side / max(image.size.height, 1))
            let width = image.size.width * ratio
            let height = image.size.height * ratio
            image.draw(in: CGRect(x: (side - width) / 2,
                                  y: (side - height) / 2,
                                  width: width, height: height))
        }
    }

    private func symbolForLabel(_ label: String) -> String? {
        guard label.count == 5, label.first == "u",
              let number = UInt32(label.dropFirst(), radix: 16),
              let scalar = UnicodeScalar(number) else { return nil }
        let symbol = String(scalar)
        // Allow only the exact 46 Arabic classes. No English digit labels.
        return spokenNames[symbol] == nil ? nil : symbol
    }

    private var spokenNames: [String: String] {
        [
            "ا": "ألف", "ب": "باء", "ت": "تاء", "ث": "ثاء",
            "ج": "جيم", "ح": "حاء", "خ": "خاء", "د": "دال",
            "ذ": "ذال", "ر": "راء", "ز": "زاي", "س": "سين",
            "ش": "شين", "ص": "صاد", "ض": "ضاد", "ط": "طاء",
            "ظ": "ظاء", "ع": "عين", "غ": "غين", "ف": "فاء",
            "ق": "قاف", "ك": "كاف", "ل": "لام", "م": "ميم",
            "ن": "نون", "ه": "هاء", "و": "واو", "ي": "ياء",
            "ء": "همزة", "آ": "ألف ممدودة", "أ": "ألف بهمزة فوق",
            "ؤ": "همزة على واو", "إ": "ألف بهمزة تحت",
            "ئ": "همزة على ياء", "ى": "ألف مقصورة", "ة": "تاء مربوطة",
            "٠": "صفر", "١": "واحد", "٢": "اثنان", "٣": "ثلاثة",
            "٤": "أربعة", "٥": "خمسة", "٦": "ستة", "٧": "سبعة",
            "٨": "ثمانية", "٩": "تسعة"
        ]
    }

    private func openCamera() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            reportError("الكاميرا غير متاحة. استخدم iPhone حقيقيًا أو اختر صورة من المعرض.")
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            showCamera = true
        case .notDetermined:
            Task { @MainActor in
                if await AVCaptureDevice.requestAccess(for: .video) {
                    showCamera = true
                } else {
                    reportCameraPermissionError()
                }
            }
        case .denied, .restricted:
            reportCameraPermissionError()
        @unknown default:
            reportError("تعذر التحقق من إذن الكاميرا.")
        }
    }

    private func reportCameraPermissionError() {
        reportError("اسمح بالكاميرا لتطبيق ArabicReader من إعدادات الجوال.")
    }

    private func reportError(_ message: String) {
        errorMessage = message
        showError = true
    }
}

// All non-Sendable Vision/Core ML objects stay inside this actor.
actor SymbolRecognitionWorker {
    private var cachedModel: VNCoreMLModel?

    func predict(_ data: Data) throws
        -> (identifier: String, confidence: Float, margin: Float) {
        let model: VNCoreMLModel
        if let cachedModel {
            model = cachedModel
        } else {
            guard let url = Bundle.main.url(forResource: "ArabicSymbols",
                                             withExtension: "mlmodelc") else {
                throw NSError(domain: "ArabicReader", code: 1, userInfo: [
                    NSLocalizedDescriptionKey:
                        "ملف النموذج غير موجود داخل التطبيق. أضف ArabicSymbols.mlmodel إلى هدف ArabicReader في Xcode."
                ])
            }
            let loadedModel = try MLModel(contentsOf: url)
            model = try VNCoreMLModel(for: loadedModel)
            cachedModel = model
        }

        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(data: data, options: [:])
        try handler.perform([request])
        guard let observations = request.results as? [VNClassificationObservation],
              !observations.isEmpty else {
            throw NSError(domain: "ArabicReader", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "لم يُرجع النموذج نتيجة تصنيف."
            ])
        }
        let sorted = observations.sorted { $0.confidence > $1.confidence }
        let first = sorted[0]
        let secondConfidence: Float = sorted.count > 1 ? sorted[1].confidence : 0
        return (first.identifier, first.confidence,
                first.confidence - secondConfidence)
    }
}

@MainActor
struct CameraPicker: UIViewControllerRepresentable {
    let onPhoto: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraDevice = .rear
        // Apple's crop screen lets the user isolate one complete symbol.
        picker.allowsEditing = true
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ picker: UIImagePickerController,
                                context: Context) { }

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    @MainActor
    final class Coordinator: NSObject, UIImagePickerControllerDelegate,
                             UINavigationControllerDelegate {
        private let parent: CameraPicker

        init(parent: CameraPicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info:
                                   [UIImagePickerController.InfoKey: Any]) {
            let photo = (info[.editedImage] as? UIImage)
                ?? (info[.originalImage] as? UIImage)
            parent.dismiss()
            if let photo { parent.onPhoto(photo) }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

#Preview {
    ContentView()
}
