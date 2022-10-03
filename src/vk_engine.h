// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>

class VulkanEngine {
public:

	bool isInitialized{ false };
	int frameNumber{ 0 };

	VkExtent2D windowExtent{ 1700 , 900 };

	struct SDL_Window* window{ nullptr };

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

	VkPipelineLayout trianglePipelineLayout;
	VkPipeline trianglePipeline;

	// Initializes everything in the engine
	void Init();

	// Shuts down the engine
	void Cleanup();

	// Draw loop
	void Draw();

	// Run main loop
	void Run();

private:
	void InitVulkan();
	void InitSwapchain();
	void InitCommands();
	void InitDefaultRenderpass();
	void InitFramebuffers();
	void InitSyncStructures();
	void InitPipelines();

	bool LoadShaderModule(const char* filePath, VkShaderModule* outShaderModule);
};

class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	VkPipelineVertexInputStateCreateInfo vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo inputAssembly;
	VkViewport viewport;
	VkRect2D scissor;
	VkPipelineRasterizationStateCreateInfo rasterizer;
	VkPipelineColorBlendAttachmentState colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo multisampling;
	VkPipelineLayout pipelineLayout;

	VkPipeline BuildPipeline(VkDevice device, VkRenderPass pass);
};