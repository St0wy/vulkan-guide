// ReSharper disable CppClangTidyClangDiagnosticUnusedMacros
// ReSharper disable CppInitializedValueIsAlwaysRewritten
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>
#include <iostream>
#include <fstream>

#include "VkBootstrap.h"

#define VK_CHECK(x)															\
	do																		\
	{																		\
		VkResult err = x;													\
		if (err)															\
		{																	\
			std::cout << "Detected Vulkan error : " << err << std::endl;	\
			abort();														\
		}																	\
	} while (0)																\

void VulkanEngine::Init()
{
	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);

	constexpr SDL_WindowFlags windowFlags = SDL_WINDOW_VULKAN;

	const int width = static_cast<int>(_windowExtent.width);
	const int height = static_cast<int>(_windowExtent.height);
	_window = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		width,
		height,
		windowFlags
	);

	InitVulkan();
	InitSwapchain();
	InitCommands();
	InitDefaultRenderpass();
	InitFramebuffers();
	InitSyncStructures();
	InitPipelines();

	//everything went fine
	_isInitialized = true;
}

void VulkanEngine::Cleanup()
{
	if (_isInitialized) {
		// Make sure the gpu has stopped doing its things
		vkDeviceWaitIdle(_device);

		vkDestroyCommandPool(_device, _commandPool, nullptr);

		// Destroy sync objects
		vkDestroyFence(_device, _renderFence, nullptr);
		vkDestroySemaphore(_device, _renderSemaphore, nullptr);
		vkDestroySemaphore(_device, _presentSemaphore, nullptr);

		vkDestroySwapchainKHR(_device, _swapchain, nullptr);

		// Destroy the main renderpass
		vkDestroyRenderPass(_device, _renderPass, nullptr);

		// Destroy swapchain resources
		for (std::size_t i = 0; i < _swapchainImageViews.size(); i++)
		{
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
		}

		vkDestroySurfaceKHR(_instance, _surface, nullptr);

		vkDestroyDevice(_device, nullptr);
		vkb::destroy_debug_utils_messenger(_instance, _debugMessenger);
		vkDestroyInstance(_instance, nullptr);

		SDL_DestroyWindow(_window);
	}
}

void VulkanEngine::Draw()
{
	// Wait until the GPU has finished rendering the last frame with a timeout of 1 second
	VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1'000'000'000));
	VK_CHECK(vkResetFences(_device, 1, &_renderFence));

	// Request image from the swapchain, one second timeout
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1'000'000'000, _presentSemaphore, nullptr, &swapchainImageIndex));

	// Now that we are sure that the command finished executing, we can safely reset the command buffer to begin recording again
	VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

	const VkCommandBuffer commandBuffer = _mainCommandBuffer;

	// Begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan know that
	const VkCommandBufferBeginInfo beginInfo = vkinit::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	// Make a clear color form frame number. This will flash with a 120 * pi frame period
	VkClearValue clearValue;
	const auto frameNumber = static_cast<float>(_frameNumber);
	const float flash = std::abs(std::sin(frameNumber / 120.0f));
	clearValue.color = { {0.0f, 0.0f, flash, 1.0f} };

	// Start the main render pass.
	// We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo renderPassBeginInfo = vkinit::RenderPassBeginInfo(_renderPass, _windowExtent, _framebuffers[swapchainImageIndex]);

	// Connect clear values
	renderPassBeginInfo.clearValueCount = 1;
	renderPassBeginInfo.pClearValues = &clearValue;

	vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	// Finalize the render pass
	vkCmdEndRenderPass(commandBuffer);
	// Finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	// Prepare the submission to the queue
	// We want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
	// We will signal the _renderSemaphore, to signal that rendering has finished

	constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submit = vkinit::SubmitInfo(&commandBuffer);
	submit.pWaitDstStageMask = &waitStage;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &_presentSemaphore;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &_renderSemaphore;

	// Submit command buffer to the queue and execute it
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

	// This will put the image we just rendered into the visible window
	// We want to wait on the _renderSemaphore for that,
	// as it's necessary that drawing command have finished before the image is displayed
	VkPresentInfoKHR presentInfo = vkinit::PresentInfo();

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &_renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	// Increase the number of frames drawn
	_frameNumber++;
}

void VulkanEngine::Run()
{
	SDL_Event e;
	bool bQuit = false;

	//main loop
	while (!bQuit)
	{
		//Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			//close the window when user alt-f4s or clicks the X button			
			if (e.type == SDL_QUIT) bQuit = true;
			if (e.type == SDL_EventType::SDL_KEYDOWN) {
				const Uint8* state = SDL_GetKeyboardState(NULL);
				if (state[SDL_SCANCODE_RETURN]) {
					std::cout << "<RETURN> is pressed.\n";
				}
				if (state[SDL_SCANCODE_RIGHT] && state[SDL_SCANCODE_UP]) {
					std::cout << "Right and Up Keys Pressed.\n";
				}
			}
		}

		Draw();
	}
}

void VulkanEngine::InitVulkan()
{
	vkb::InstanceBuilder builder;

	auto instanceRet = builder.set_app_name("Example Vulkan Application")
		.request_validation_layers(true)
		.require_api_version(1, 1, 0)
		.use_default_debug_messenger()
		.build();

	const vkb::Instance vkbInstance = instanceRet.value();

	// Store the instance
	_instance = vkbInstance.instance;

	// Store the debug messenger
	_debugMessenger = vkbInstance.debug_messenger;

	// Get the surface of the window we opened with SDL
	SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

	// Use VkBootstrap to select a GPU
	// We want a GPU that can write to the SDL surface and supports Vulkan 1.1
	vkb::PhysicalDeviceSelector selector{ vkbInstance };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 1)
		.set_surface(_surface)
		.select()
		.value();

	// Create the final Vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	// ReSharper disable once CppUseStructuredBinding
	vkb::Device vkbDevice = deviceBuilder.build().value();

	// Get the VkDevice handle used in the rest of a Vulkan application
	_device = vkbDevice.device;
	_chosenGpu = physicalDevice.physical_device;

	// Use VkBootstrap to get a Graphics queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
}

void VulkanEngine::InitSwapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{ _chosenGpu, _device, _surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		.build()
		.value();

	// Store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;
	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();
	_swapchainImageFormat = vkbSwapchain.image_format;
}

void VulkanEngine::InitCommands()
{
	// Create a command pool for commands submitted to the graphics queue
	// We also want the pool to allow for resetting of individual command buffers
	const VkCommandPoolCreateInfo commandPoolInfo = vkinit::CommandPoolCreateInfo(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

	// Allocate the default command buffer that we will use for rendering
	const VkCommandBufferAllocateInfo commandAllocateInfo = vkinit::CommandBufferAllocateInfo(_commandPool, 1);

	VK_CHECK(vkAllocateCommandBuffers(_device, &commandAllocateInfo, &_mainCommandBuffer));
}

void VulkanEngine::InitDefaultRenderpass()
{
	// The renderpass will use this color attachment
	VkAttachmentDescription colorAttachment{};

	// The attachment will have the format needed by the swapchain
	colorAttachment.format = _swapchainImageFormat;

	// 1 sample, we won't be doing MSAA
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

	// We Clear when this attachment is loaded
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

	// We keep the attachment stored when the renderpass ends
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

	// We don't care about stencil
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	// We don't know or care about the starting layout of the attachment
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	// After the renderpass ends, the image has to be on a layout ready for display
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef{};

	// Attachment number will index into the pAttachments array int he parent renderpass itself
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// We are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	// Connect the attachment to the info
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;

	// Connect the subpass to the info
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	VK_CHECK(vkCreateRenderPass(_device, &renderPassInfo, nullptr, &_renderPass));
}

void VulkanEngine::InitFramebuffers()
{
	// Create the framebuffers for the swapchain images
	// This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo framebufferInfo = vkinit::FramebufferCreateInfo(_renderPass, _windowExtent);

	// Grab how many images we have in the swapchain
	const std::size_t swapchainImageCount = _swapchainImages.size();
	_framebuffers = std::vector<VkFramebuffer>(swapchainImageCount);

	// Create framebuffers for each of the swapchain image views

	for (std::size_t i = 0; i < swapchainImageCount; ++i)
	{
		framebufferInfo.pAttachments = &_swapchainImageViews[i];
		VK_CHECK(vkCreateFramebuffer(_device, &framebufferInfo, nullptr, &_framebuffers[i]));
	}
}

void VulkanEngine::InitSyncStructures()
{
	// Create synchronization structures
	const VkFenceCreateInfo fenceCreateInfo = vkinit::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);

	VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

	// For the semaphores we don't need any flags
	const VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::SemaphoreCreateInfo();

	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
	VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));
}

void VulkanEngine::InitPipelines()
{
	VkShaderModule triangleFragShader;
	if(!LoadShaderModule("../../shaders/triangle.frag.spv", &triangleFragShader))
	{
		std::cout << "Error when building the triangle fragment shader module\n";
	} else
	{
		std::cout << "Triangle fragment shader successfully loaded\n";
	}

	VkShaderModule triangleVertexShader;
	if(!LoadShaderModule("../../shaders/triangle.vert.spv", &triangleVertexShader))
	{
		std::cout << "Error when building the triangle vertex shader module\n";
	} else
	{
		std::cout << "Triangle vertex shader successfully loaded\n";
	}
}

bool VulkanEngine::LoadShaderModule(const char* filePath, VkShaderModule* outShaderModule)
{
	// Open the file with a cursor at the end
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) return false;

	// Find what the size of the file is using the cursor
	// Because the cursor is at the end, it gives the size directly in bytes
	const std::size_t fileSize = file.tellg();

	// spirv expects the buffer to be on uint32, so make sure to reserve an int vector big enough for the entire file
	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	// Put file cursor at the beginning
	file.seekg(0);

	// Load file in buffer
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

	// Now that the file is loaded into the buffer, we can close it
	file.close();

	// Create a new shader module, using the buffer we loaded
	VkShaderModuleCreateInfo createInfo = vkinit::ShaderModuleCreateInfo();

	// codeSize has to be in bytes, so multiply by the ints in the buffer by size of int to know the real size of the buffer
	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	// Check that the creation goes well
	VkShaderModule shaderModule;
	if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		return false;
	}

	*outShaderModule = shaderModule;
	return true;
}
