// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <deque>
#include <functional>
#include <vector>

#include "vk_mesh.h"
#include "vk_types.h"

struct DeletionQueue
{
    std::deque<std::function<void()>> deletors;

    void PushFunction(std::function<void()>&& function);

    void Flush();
};

class VulkanEngine
{
public:
    bool isInitialized{false};
    int frameNumber{0};

    VkExtent2D windowExtent{1200, 700};

    struct SDL_Window* window{nullptr};

    VkInstance instance{};
    VkDebugUtilsMessengerEXT debugMessenger{};
    VkPhysicalDevice chosenGpu;
    VkDevice device;
    VkSurfaceKHR surface;

    VkSwapchainKHR swapchain;
    VkFormat swapchainImageFormat;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;

    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

    VkRenderPass renderPass;

    std::vector<VkFramebuffer> framebuffers;

    VkSemaphore presentSemaphore, renderSemaphore;
    VkFence renderFence;

    //VkPipelineLayout trianglePipelineLayout;
    VkPipelineLayout meshPipelineLayout;

    VkPipeline meshPipeline;
    Mesh triangleMesh;

    VmaAllocator allocator;


    // Initializes everything in the engine
    void Init();

    // Shuts down the engine
    void Cleanup();

    // Draw loop
    void Draw();

    // Run main loop
    void Run();

private:
    int _selectedShader{0};
    DeletionQueue _mainDeletionQueue;

    void InitVulkan();
    void InitSwapchain();
    void InitCommands();
    void InitDefaultRenderpass();
    void InitFramebuffers();
    void InitSyncStructures();
    void InitPipelines();

    bool LoadShaderModule(const char* filePath, VkShaderModule* outShaderModule);

    void LoadMeshes();
    void UploadMesh(Mesh& mesh);
};

class PipelineBuilder
{
public:
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    VkViewport viewport{};
    VkRect2D scissor{};
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    VkPipelineMultisampleStateCreateInfo multisampling{};
    VkPipelineLayout pipelineLayout{};

    VkPipeline BuildPipeline(VkDevice device, VkRenderPass pass) const;
};
