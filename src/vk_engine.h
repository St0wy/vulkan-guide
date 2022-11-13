// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "camera.h"
#include "vk_mesh.h"
#include "vk_types.h"

constexpr std::size_t FRAME_OVERLAP = 2;

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void PushFunction(std::function<void()>&& function);

	void Flush();
};

class VulkanEngine
{
public:
	bool isInitialized{ false };
	int frameNumber{ 0 };

	VkExtent2D windowExtent{ 1200, 700 };

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

	VkRenderPass renderPass;

	std::vector<VkFramebuffer> framebuffers;
	
	VmaAllocator allocator;

	VkImageView depthImageView;
	AllocatedImage depthImage;
	VkFormat depthFormat;

	std::vector<RenderObject> renderables;
	std::unordered_map<std::string, Material> materials;
	std::unordered_map<std::string, Mesh> meshes;

	FrameData frames[FRAME_OVERLAP];

	VkDescriptorSetLayout globalSetLayout;
	VkDescriptorSetLayout objectSetLayout;
	VkDescriptorPool descriptorPool;

	VkPhysicalDeviceProperties gpuProperties;

	GpuSceneData sceneParameters;
	AllocatedBuffer sceneParameterBuffer;

	// Initializes everything in the engine
	void Init();

	// Shuts down the engine
	void Cleanup();

	// Draw loop
	void Draw();

	// Run main loop
	void Run();

	bool Update();

	Material* CreateMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);

	Material* GetMaterial(const std::string& name);

	Mesh* GetMesh(const std::string& name);

	void DrawObjects(VkCommandBuffer commandBuffer, RenderObject* first, int count);

	FrameData& GetCurrentFrame();

	[[nodiscard]] AllocatedBuffer CreateBuffer(std::size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) const;

	[[nodiscard]] std::size_t PadUniformBufferSize(std::size_t originalSize) const;

private:
	int _selectedShader{ 0 };
	DeletionQueue _mainDeletionQueue;
	Camera _camera{ { 0.0f, 6.0f, 10.0f }, 1700.0f / 900.0f };
	bool _hasFocus = true;

	void InitVulkan();
	void InitSwapchain();
	void InitCommands();
	void InitDefaultRenderpass();
	void InitFramebuffers();
	void InitSyncStructures();
	void InitPipelines();
	void InitScene();
	void InitDescriptors();

	bool LoadShaderModule(const char* filePath, VkShaderModule* outShaderModule) const;

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
	VkPipelineDepthStencilStateCreateInfo depthStencil{};

	VkPipeline BuildPipeline(VkDevice device, VkRenderPass pass) const;
};
