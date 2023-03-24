//
//  ImageGenerator.swift
//  SD2Img2Img
//
//  Created by HanGyo Jeong on 2023/03/23.
//

import Foundation
import StableDiffusion
import CoreML
import UIKit

// @MainActor: MainThread에서의 동작을 보장
@MainActor
final class ImageGenerator: ObservableObject {
    
    struct GenerationParameter {
        var prompt: String
        var negativePrompt: String
        var guidanceScale: Float
        var seed: Int
        var stepCount: Int
        var imageCount: Int
        var disableSafety: Bool
        var startImage: CGImage?
        var strength: Float = 1.0
    }
    
    struct GeneratedImage: Identifiable {
        let id: UUID = UUID()
        let uiImage: UIImage
    }
    
    struct GeneratedImages {
        let prompt: String
        let negativePrompt: String
        let guidanceScale: Float
        let imageCount: Int
        let stepCount: Int
        let seed: Int
        let disableSafety: Bool
        let images: [GeneratedImage]
    }
    
    enum GenerationState: Equatable {
        case idle
        case generating(progressStep: Int)
        static func == (lhs: Self, rhs: Self) -> Bool {
            switch(lhs, rhs){
            case (.idle, idle):
                return true
            case (.generating(let step1), .generating(let step2)):
                if step1 == step2 {
                    return true
                } else {
                    return false
                }
            default:
                return false
            }
        
        }
    }
    
    @Published var generationState: GenerationState = .idle
    @Published var generatedImages: GeneratedImages?
    @Published var isPipelineCreated = false
    
    private var sdPipeline: StableDiffusionPipeline?
    
    init() {
        
    }
    
    // MARK: Setter Funcs
    func setState(_ state: GenerationState) {
        generationState = state
    }
    
    func setPipeline(_ pipeline: StableDiffusionPipeline) {
        sdPipeline = pipeline
        isPipelineCreated = true
    }
    
    func setGeneratedImages(_ images: GeneratedImages) {
        generatedImages = images
    }
    
    // swiftlint:disable function_body_length
    func generateImages(_ parameter: GenerationParameter) {
        guard generationState == .idle else { return }
        // Runs the given nonthrowing operation asynchronously as part of a new top-level task.
        Task.detached(priority: .high) {
            await self.setState(.generating(progressStep: 0))
            
            if await self.sdPipeline == nil {
                guard let path = Bundle.main.path(forResource: "CoreMLModels", ofType: nil, inDirectory: nil) else {
                    fatalError("Fatal error: failed to find the CoreML Models.")
                }
                let resourceURL = URL(fileURLWithPath: path)
                
                let config = MLModelConfiguration()
                
                /*
                 [Note]
                 Specifying config.computeUnits is not necessary. Use the default
                 
                 Specifying config.computeUnits = .cpuAndNeuralEngine will cause an internal fatal error on devices.
                 config.computeUnits = .cpuAndNeuralEngine
                 
                 Specifying config.computeUnits = .cpuAndGPU works on device with no reason
                 if !ProcessInfo.processInfo.isiOSAppOnMac {
                    config.computeUnits = .cpuAndGPU
                 }
                 */
                
                // ReduceMemory option was added at v0.1.0
                // On iOS, the reduceMemory option should be set to true
                let reduceMemory = ProcessInfo.processInfo.isiOSAppOnMac ? false:true
                if let pipeline = try? StableDiffusionPipeline(resourcesAt: resourceURL, configuration: config, reduceMemory: reduceMemory) {
                    await self.setPipeline(pipeline)
                } else {
                    fatalError("Fatal error: failed to create the Stable-Diffusion-Pipeline.")
                }
            }
            
            if let sdPipeline = await self.sdPipeline {
                do {
                    // Will Add ProgressHandle
                    
                    var configuration = StableDiffusionPipeline.Configuration(prompt: parameter.prompt)
                    configuration.negativePrompt = parameter.negativePrompt
                    configuration.imageCount = parameter.imageCount
                    configuration.stepCount = parameter.stepCount
                    configuration.seed = UInt32(parameter.seed)
                    configuration.guidanceScale = parameter.guidanceScale
                    configuration.disableSafety = parameter.disableSafety
                    
                    configuration.startingImage = parameter.startImage
                    configuration.strength = parameter.strength
                    
                    let cgImages = try sdPipeline.generateImages(configuration: configuration)
                    print("Images were Generated")
                    
                    let uiImages = cgImages.compactMap { image in
                        if let cgImage = image {
                            return UIImage(cgImage: cgImage)
                        } else {
                            return nil
                        }
                    }
                    
                    await self.setGeneratedImages(GeneratedImages(prompt: parameter.prompt,
                                                                  negativePrompt: parameter.negativePrompt,
                                                                  guidanceScale: parameter.guidanceScale,
                                                                  imageCount: parameter.imageCount,
                                                                  stepCount: parameter.stepCount,
                                                                  seed: parameter.seed,
                                                                  disableSafety: parameter.disableSafety,
                                                                  images: uiImages.map{ uiImage in GeneratedImage(uiImage: uiImage) }))
                } catch {
                    print("Failed to generate images.")
                }
            }
        }
    }
}
