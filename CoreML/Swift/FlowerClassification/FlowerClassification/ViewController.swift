//
//  ViewController.swift
//  FlowerClassification
//
//  Created by HanGyo Jeong on 2023/03/23.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {
    let NUM_CLASSES = 5
    var currentImage = 1
    
    @IBOutlet weak var txtOutput: UILabel!
    @IBOutlet weak var imageView: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }

    func interpretImage() {
        let theImage: UIImage = UIImage(named: String(currentImage))!
        getClassification(for: theImage)
    }
    
    @IBAction func prevButton(_ sender: Any) {
        currentImage = currentImage - 1
        if currentImage <= 0 {
            currentImage = 6
        }
        loadImage()
    }
    @IBAction func nextButton(_ sender: Any) {
        currentImage = currentImage + 1
        if currentImage >= 7 {
            currentImage = 1
        }
        loadImage()
    }
    @IBAction func classifyButton(_ sender: Any) {
        interpretImage()
    }
    
    func loadImage(){
            imageView.image = UIImage(named: String(currentImage))
        }
    
    func getClassification(for image: UIImage) {
        let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue))!
        guard let ciImage = CIImage(image: image) else { fatalError("...") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                print("...")
            }
        }
    }
    
    // VNCoreMLRequest는 내부적으로 모델 초기화를 함
    lazy var classificationRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel.init(for: flowers().model)
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processResults(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("...")
        }
    }()
    
    func processResults(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.txtOutput.text = "..."
                return
            }
            let classifications = results as! [VNClassificationObservation]
            
            if classifications.isEmpty {
                self.txtOutput.text = "Nothing recognized."
            } else {
                let topClassifications = classifications.prefix(self.NUM_CLASSES)
                let descriptions = topClassifications.map {
                    classification in return String(format: " (%.2f) %@", classification.confidence, classification.identifier)
                }
                self.txtOutput.text = "Classification:\n" + descriptions.joined(separator: "\n")
            }
        }
    }
}

