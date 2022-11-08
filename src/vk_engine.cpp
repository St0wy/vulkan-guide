// ReSharper disable CppClangTidyClangDiagnosticUnusedMacros
// ReSharper disable CppClangTidyClangDiagnosticUnusedMacros
// ReSharper disable CppInitializedValueIsAlwaysRewritten
#include "vk_engine.h"

#include <fstream>
#include <iostream>
#include <ranges>
#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "vk_initializers.h"
#include "vk_types.h"


#define VK_CHECK(x)															\
	do																		\
	{																		\
		VkResult err = x;													\
		if (err)															\
		{																	\
			std::cerr << "Detected Vulkan error : " << err << std::endl;	\
			abort();														\
		}																	\
	} while (0)

void DeletionQueue::PushFunction(std::function<void()>&& function)
{
	deletors.push_back(function);
}

void DeletionQueue::Flush()
{
	// Reverse iterate the deletion queue to execute all the functions
	for (auto& deletor : std::ranges::reverse_view(deletors))
	{
		deletor();
	}

	deletors.clear();
}

void VulkanEngine::Init()
{
	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);

	constexpr SDL_WindowFlags windowFlags = SDL_WINDOW_VULKAN;

	const int width = static_cast<int>(windowExtent.width);
	const int height = static_cast<int>(windowExtent.height);
	window = SDL_CreateWindow(
		"Vulkan Engine",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		width,
		height,
		windowFlags
	);

	SDL_SetRelativeMouseMode(SDL_TRUE);

	InitVulkan();
	InitSwapchain();
	InitCommands();
	InitDefaultRenderpass();
	InitFramebuffers();
	InitSyncStructures();
	InitDescriptors();
	InitPipelines();

	LoadMeshes();

	InitScene();

	//everything went fine
	isInitialized = true;
}

void VulkanEngine::Cleanup()
{
	if (!isInitialized) return;

	// Make sure the gpu has stopped doing its things
	vkWaitForFences(device, 1, &GetCurrentFrame().renderFence, true, 1'000'000'000);

	_mainDeletionQueue.Flush();

	vkDestroySurfaceKHR(instance, surface, nullptr);

	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);

	SDL_DestroyWindow(window);
}

void VulkanEngine::Draw()
{
	// Wait until the GPU has finished rendering the last frame with a timeout of 1 second
	VK_CHECK(vkWaitForFences(device, 1, &GetCurrentFrame().renderFence, true, 1'000'000'000));
	VK_CHECK(vkResetFences(device, 1, &GetCurrentFrame().renderFence));

	// Request image from the swapchain, one second timeout
	uint32_t swapchainImageIndex;
	VK_CHECK(
		vkAcquireNextImageKHR(device, swapchain, 1'000'000'000, GetCurrentFrame().presentSemaphore, nullptr, &swapchainImageIndex));

	// Now that we are sure that the command finished executing, we can safely reset the command buffer to begin recording again
	VK_CHECK(vkResetCommandBuffer(GetCurrentFrame().mainCommandBuffer, 0));

	// ReSharper disable once CppLocalVariableMayBeConst
	VkCommandBuffer commandBuffer = GetCurrentFrame().mainCommandBuffer;

	// Begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan know that
	const VkCommandBufferBeginInfo beginInfo = vkinit::CommandBufferBeginInfo(
		VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	// Make a clear color form frame number. This will flash with a 120 * pi frame period
	VkClearValue clearValue{};
	//const auto floatFrameNumber = static_cast<float>(frameNumber);
	//const float flash = std::abs(std::sin(floatFrameNumber / 120.0f));
	clearValue.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

	VkClearValue depthClear{};
	depthClear.depthStencil.depth = 1.0f;

	// Start the main render pass.
	// We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo renderPassBeginInfo = vkinit::RenderPassBeginInfo(
		renderPass,
		windowExtent,
		framebuffers[swapchainImageIndex]);

	// Connect clear values
	const VkClearValue clearValues[2] = { clearValue, depthClear };
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = &clearValues[0];

	vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	DrawObjects(commandBuffer, renderables.data(), static_cast<int>(renderables.size()));

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
	submit.pWaitSemaphores = &GetCurrentFrame().presentSemaphore;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &GetCurrentFrame().renderSemaphore;

	// Submit command buffer to the queue and execute it
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submit, GetCurrentFrame().renderFence));

	// This will put the image we just rendered into the visible window
	// We want to wait on the _renderSemaphore for that,
	// as it's necessary that drawing command have finished before the image is displayed
	VkPresentInfoKHR presentInfo = vkinit::PresentInfo();

	presentInfo.pSwapchains = &swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &GetCurrentFrame().renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	VK_CHECK(vkQueuePresentKHR(graphicsQueue, &presentInfo));

	// Increase the number of frames drawn
	frameNumber++;
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
			if (e.type == SDL_QUIT)
			{
				bQuit = true;
			}
			else if (e.type == SDL_KEYDOWN)
			{
				const Uint8* state = SDL_GetKeyboardState(nullptr);
				if (state[SDL_SCANCODE_RETURN])
				{
					std::cout << "<RETURN> is pressed.\n";
				}

				if (state[SDL_SCANCODE_RIGHT] && state[SDL_SCANCODE_UP])
				{
					std::cout << "Right and Up Keys Pressed.\n";
				}

				if (e.key.keysym.sym == SDLK_SPACE)
				{
					_selectedShader++;
					_selectedShader %= 2;
				}
			}
			else if (e.type == SDL_WINDOWEVENT_FOCUS_GAINED)
			{
				_hasFocus = true;
			}
			else if (e.type == SDL_WINDOWEVENT_FOCUS_LOST)
			{
				_hasFocus = false;
			}
			else if (e.type == SDL_MOUSEMOTION)
			{
				const auto xMotion = static_cast<float>(e.motion.xrel);
				const auto yMotion = static_cast<float>(e.motion.yrel);
				_camera.SetYaw(_camera.Yaw() + -glm::radians(xMotion) * 0.3f);
				_camera.SetPitch(_camera.Pitch() + -glm::radians(yMotion) * 0.3f);
			}
		}

		if (_hasFocus)
		{
			const bool shouldQuit = Update();
			if (shouldQuit)
			{
				bQuit = true;
			}
		}

		Draw();
	}
}

bool VulkanEngine::Update()
{
	const Uint8* keyboardState = SDL_GetKeyboardState(nullptr);

	if (keyboardState[SDL_SCANCODE_ESCAPE])
	{
		return true;
	}

	constexpr float camSpeed = 0.1f;
	if (keyboardState[SDL_SCANCODE_W])
	{
		_camera.position += _camera.Front() * camSpeed;
	}
	if (keyboardState[SDL_SCANCODE_A])
	{
		_camera.position += _camera.Right() * -camSpeed;
	}
	if (keyboardState[SDL_SCANCODE_S])
	{
		_camera.position += _camera.Front() * -camSpeed;
	}
	if (keyboardState[SDL_SCANCODE_D])
	{
		_camera.position += _camera.Right() * camSpeed;
	}
	if (keyboardState[SDL_SCANCODE_LSHIFT])
	{
		_camera.position += glm::vec3{ 0, -camSpeed, 0 };
	}
	if (keyboardState[SDL_SCANCODE_SPACE])
	{
		_camera.position += glm::vec3{ 0, camSpeed, 0 };
	}

	return false;
}

Material* VulkanEngine::CreateMaterial(const VkPipeline pipeline, const VkPipelineLayout layout,
	const std::string& name)
{
	Material material{};
	material.pipeline = pipeline;
	material.pipelineLayout = layout;
	materials[name] = material;
	return &materials[name];
}

Material* VulkanEngine::GetMaterial(const std::string& name)
{
	const auto it = materials.find(name);

	if (it == materials.end()) return nullptr;

	return &(*it).second;
}

Mesh* VulkanEngine::GetMesh(const std::string& name)
{
	const auto it = meshes.find(name);

	if (it == meshes.end()) return nullptr;

	return &(*it).second;
}

void VulkanEngine::DrawObjects(const VkCommandBuffer commandBuffer, RenderObject* first, const int count)
{
	const glm::mat4 view = _camera.GetViewMatrix();

	const glm::mat4 projection = _camera.GetProjectionMatrix();

	const Mesh* lastMesh = nullptr;
	const Material* lastMaterial = nullptr;
	for (int i = 0; i < count; i++)
	{
		const auto& [mesh, material, transformMatrix] = first[i];

		if (!mesh || !material) continue;

		// Only bind the pipeline if it doesn't match with the already bound one
		if (material != lastMaterial)
		{
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material->pipeline);
			lastMaterial = material;
		}

		glm::mat4 model = transformMatrix;

		// Final render matrix, that we are computing on the cpu
		const glm::mat4 meshMatrix = projection * view * model;

		MeshPushConstants constants{};
		constants.renderMatrix = meshMatrix;

		// Upload the mesh to the GPU via push constants
		vkCmdPushConstants(commandBuffer, material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

		// Only bind the mesh if it's a different one from last bind
		if (mesh != lastMesh)
		{
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mesh->vertexBuffer.buffer, &offset);
			lastMesh = mesh;
		}

		// We can now draw
		vkCmdDraw(commandBuffer, static_cast<uint32_t>(mesh->vertices.size()), 1, 0, 0);
	}
}

FrameData& VulkanEngine::GetCurrentFrame()
{
	return frames[frameNumber % FRAME_OVERLAP];
}

AllocatedBuffer VulkanEngine::CreateBuffer(const std::size_t allocSize, const VkBufferUsageFlags usage, const VmaMemoryUsage memoryUsage) const
{
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;
	bufferInfo.size = allocSize;
	bufferInfo.usage = usage;

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = memoryUsage;

	AllocatedBuffer newBuffer{};

	VK_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, nullptr));

	return newBuffer;
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
	instance = vkbInstance.instance;

	// Store the debug messenger
	debugMessenger = vkbInstance.debug_messenger;

	// Get the surface of the window we opened with SDL
	SDL_Vulkan_CreateSurface(window, instance, &surface);

	// Use VkBootstrap to select a GPU
	// We want a GPU that can write to the SDL surface and supports Vulkan 1.1
	vkb::PhysicalDeviceSelector selector{ vkbInstance };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 1)
		.set_surface(surface)
		.select()
		.value();

	// Create the final Vulkan device
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	// ReSharper disable once CppUseStructuredBinding
	vkb::Device vkbDevice = deviceBuilder.build().value();

	// Get the VkDevice handle used in the rest of a Vulkan application
	device = vkbDevice.device;
	chosenGpu = physicalDevice.physical_device;

	// Use VkBootstrap to get a Graphics queue
	graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	// Init the memory allocator
	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = chosenGpu;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;
	vmaCreateAllocator(&allocatorInfo, &allocator);

	_mainDeletionQueue.PushFunction([this]()
	{
		vmaDestroyAllocator(allocator);
	});
}

void VulkanEngine::InitSwapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{ chosenGpu, device, surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(windowExtent.width, windowExtent.height)
		.build()
		.value();

	// Store swapchain and its related images
	swapchain = vkbSwapchain.swapchain;
	swapchainImages = vkbSwapchain.get_images().value();
	swapchainImageViews = vkbSwapchain.get_image_views().value();
	swapchainImageFormat = vkbSwapchain.image_format;

	_mainDeletionQueue.PushFunction([this]()
	{
		vkDestroySwapchainKHR(device, swapchain, nullptr);
	});

	// Depth image will match the window
	VkExtent3D depthImageExtent = {
		windowExtent.width,
		windowExtent.height,
		1
	};

	// Hardcoding the depth format to 32 bit float
	depthFormat = VK_FORMAT_D32_SFLOAT;

	// The depth image will be an image with the format we selected and Depth Attachment usage flag
	VkImageCreateInfo depthImageInfo = vkinit::ImageCreateInfo(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		depthImageExtent);

	// For the depth image, we want to allocate it from GPU local memory
	VmaAllocationCreateInfo depthImageAllocInfo{};
	depthImageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	depthImageAllocInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	// Allocate and create the image
	vmaCreateImage(allocator, &depthImageInfo, &depthImageAllocInfo, &depthImage.image, &depthImage.allocation,
		nullptr);

	// Build an image-view for the depth image to use for rendering
	VkImageViewCreateInfo depthViewInfo = vkinit::ImageViewCreateInfo(depthFormat, depthImage.image,
		VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(device, &depthViewInfo, nullptr, &depthImageView));

	// Add to deletion queues
	_mainDeletionQueue.PushFunction([this]()
	{
		vkDestroyImageView(device, depthImageView, nullptr);
		vmaDestroyImage(allocator, depthImage.image, depthImage.allocation);
	});
}

void VulkanEngine::InitCommands()
{
	// Create a command pool for commands submitted to the graphics queue
	// We also want the pool to allow for resetting of individual command buffers
	const VkCommandPoolCreateInfo commandPoolInfo = vkinit::CommandPoolCreateInfo(graphicsQueueFamily,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	for (auto& frame : frames)
	{
		VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frame.commandPool));

		// Allocate the default command buffer that we will use for rendering
		const VkCommandBufferAllocateInfo commandAllocateInfo = vkinit::CommandBufferAllocateInfo(frame.commandPool, 1);

		VK_CHECK(vkAllocateCommandBuffers(device, &commandAllocateInfo, &frame.mainCommandBuffer));

		_mainDeletionQueue.PushFunction([this, &frame]()
		{
			vkDestroyCommandPool(device, frame.commandPool, nullptr);
		});
	}
}

void VulkanEngine::InitDefaultRenderpass()
{
	// The renderpass will use this color attachment
	VkAttachmentDescription colorAttachment{};

	// The attachment will have the format needed by the swapchain
	colorAttachment.format = swapchainImageFormat;

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

	VkAttachmentDescription depthAttachment{};
	depthAttachment.flags = 0;
	depthAttachment.format = depthFormat;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentRef{};
	depthAttachmentRef.attachment = 1;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// We are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	subpass.pDepthStencilAttachment = &depthAttachmentRef;

	VkAttachmentDescription attachments[2] = { colorAttachment, depthAttachment };

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	// Connect the attachment to the info
	renderPassInfo.attachmentCount = 2;
	renderPassInfo.pAttachments = attachments;

	// Connect the subpass to the info
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency depthDependency = {};
	depthDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	depthDependency.dstSubpass = 0;
	depthDependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
		VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depthDependency.srcAccessMask = 0;
	depthDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
		VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depthDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency dependencies[2] = { dependency, depthDependency };

	renderPassInfo.dependencyCount = 2;
	renderPassInfo.pDependencies = dependencies;

	VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));

	_mainDeletionQueue.PushFunction([this]()
	{
		vkDestroyRenderPass(device, renderPass, nullptr);
	});
}

void VulkanEngine::InitFramebuffers()
{
	// Create the framebuffers for the swapchain images
	// This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo framebufferInfo = vkinit::FramebufferCreateInfo(renderPass, windowExtent);

	// Grab how many images we have in the swapchain
	const std::size_t swapchainImageCount = swapchainImages.size();
	framebuffers = std::vector<VkFramebuffer>(swapchainImageCount);

	// Create framebuffers for each of the swapchain image views

	for (std::size_t i = 0; i < swapchainImageCount; ++i)
	{
		VkImageView attachments[2]{};
		attachments[0] = swapchainImageViews[i];
		attachments[1] = depthImageView;
		framebufferInfo.attachmentCount = 2;
		framebufferInfo.pAttachments = attachments;
		VK_CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]));

		_mainDeletionQueue.PushFunction([this, i]()
		{
			vkDestroyFramebuffer(device, framebuffers[i], nullptr);
			vkDestroyImageView(device, swapchainImageViews[i], nullptr);
		});
	}
}

void VulkanEngine::InitSyncStructures()
{
	// Create synchronization structures
	const VkFenceCreateInfo fenceCreateInfo = vkinit::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);

	// For the semaphores we don't need any flags
	const VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::SemaphoreCreateInfo();

	for (auto& frame : frames)
	{
		VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frame.renderFence));

		_mainDeletionQueue.PushFunction([this, &frame]()
		{
			vkDestroyFence(device, frame.renderFence, nullptr);
		});

		VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.presentSemaphore));
		VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.renderSemaphore));

		_mainDeletionQueue.PushFunction([this, &frame]()
		{
			vkDestroySemaphore(device, frame.presentSemaphore, nullptr);
			vkDestroySemaphore(device, frame.renderSemaphore, nullptr);
		});
	}
}

void VulkanEngine::InitPipelines()
{
	// Compile colored triangle shaders
	VkShaderModule triangleFragShader;
	if (!LoadShaderModule("../../shaders/colored_triangle.frag.spv", &triangleFragShader))
	{
		std::cerr << "Error when building the triangle fragment shader module\n";
	}
	else
	{
		std::cout << "Triangle fragment shader successfully loaded\n";
	}

	// Compile mesh vertex shader
	VkShaderModule meshVertexShader;
	if (!LoadShaderModule("../../shaders/tri_mesh.vert.spv", &meshVertexShader))
	{
		std::cerr << "Error when building the triangle vertex shader module\n";
	}
	else
	{
		std::cout << "Mesh vertex shader successfully loaded\n";
	}


	// Build the stage-create-info for both vertex and fragment stages
	// This lets the pipeline know the shader modules per stage
	PipelineBuilder pipelineBuilder;
	pipelineBuilder.shaderStages.push_back(
		vkinit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVertexShader));
	pipelineBuilder.shaderStages.push_back(
		vkinit::PipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

	// Vertex input controls how to read vertices from vertex buffers
	pipelineBuilder.vertexInputInfo = vkinit::VertexInputStateCreateInfo();

	// Input assembly is the configuration for drawing triangle lists, strips, or individual points
	// We are just going to draw triangle list
	pipelineBuilder.inputAssembly = vkinit::InputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	// Build viewport and scissor from the swapchain extents
	pipelineBuilder.viewport.x = 0.0f;
	pipelineBuilder.viewport.y = 0.0f;
	pipelineBuilder.viewport.width = static_cast<float>(windowExtent.width);
	pipelineBuilder.viewport.height = static_cast<float>(windowExtent.height);
	pipelineBuilder.viewport.minDepth = 0.0f;
	pipelineBuilder.viewport.maxDepth = 1.0f;

	pipelineBuilder.scissor.offset = { 0, 0 };
	pipelineBuilder.scissor.extent = windowExtent;

	// Configure the rasterizer to draw filled triangles
	pipelineBuilder.rasterizer = vkinit::RasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);

	// We don't use multisampling, so just run the default one
	pipelineBuilder.multisampling = vkinit::MultisamplingStateCreateInfo();

	// A single blend attachment with no blending and writing to RGBA
	pipelineBuilder.colorBlendAttachment = vkinit::ColorBlendAttachmentState();

	auto [bindings, attributes, flags] = Vertex::GetVertexDescription();

	// Connect the pipeline builder vertex input info to the one we get from Vertex
	pipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = attributes.data();
	pipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());

	pipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = bindings.data();
	pipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());

	pipelineBuilder.depthStencil = vkinit::DepthStencilStateCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	VkPipelineLayoutCreateInfo meshPipelineLayoutInfo = vkinit::PipelineLayoutCreateInfo();

	// Setup push constants
	VkPushConstantRange pushConstant{};

	// This push constant starts at the beginning
	pushConstant.offset = 0;

	// This push constant takes up the size of the MeshPushConstants struct
	pushConstant.size = sizeof(MeshPushConstants);

	// This push constant range is only accessible in the vertex shader
	pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	meshPipelineLayoutInfo.pPushConstantRanges = &pushConstant;
	meshPipelineLayoutInfo.pushConstantRangeCount = 1;

	VkPipelineLayout meshPipelineLayout;
	VK_CHECK(vkCreatePipelineLayout(device, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout));

	pipelineBuilder.pipelineLayout = meshPipelineLayout;

	VkPipeline meshPipeline = pipelineBuilder.BuildPipeline(device, renderPass);

	CreateMaterial(meshPipeline, meshPipelineLayout, "defaultmesh");

	vkDestroyShaderModule(device, meshVertexShader, nullptr);
	vkDestroyShaderModule(device, triangleFragShader, nullptr);

	_mainDeletionQueue.PushFunction([=, this]()
	{
		// Destroy the pipeline we have created
		vkDestroyPipeline(device, meshPipeline, nullptr);

		vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
	});
}

void VulkanEngine::InitScene()
{
	RenderObject monkey{};
	monkey.mesh = GetMesh("monkey");
	monkey.material = GetMaterial("defaultmesh");
	monkey.transformMatrix = glm::mat4{ 1.0f };

	renderables.push_back(monkey);

	for (int x = -20; x <= 20; ++x)
	{
		for (int y = -20; y <= 20; ++y)
		{
			RenderObject monkeyArray{};
			monkeyArray.mesh = GetMesh("monkey");
			monkeyArray.material = GetMaterial("defaultmesh");
			const glm::mat4 translation = translate(glm::mat4{ 1.0f }, glm::vec3(static_cast<float>(x) * 3.0f, 2.0f, static_cast<float>(y) * 3.0f));
			monkeyArray.transformMatrix = translation;

			renderables.push_back(monkeyArray);
		}
	}

	for (int x = -30; x <= 30; ++x)
	{
		for (int y = -30; y <= 30; ++y)
		{
			RenderObject triangle{};
			triangle.mesh = GetMesh("triangle");
			triangle.material = GetMaterial("defaultmesh");
			glm::mat4 translation = translate(glm::mat4{ 1.0 }, glm::vec3(x, 0, y));
			glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2, 0.2, 0.2));
			triangle.transformMatrix = translation * scale;

			renderables.push_back(triangle);
		}
	}
}

void VulkanEngine::InitDescriptors()
{
	for (auto& frame : frames)
	{
		frame.cameraBuffer = CreateBuffer(sizeof(GpuCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	}

	for (auto& frame : frames)
	{
		_mainDeletionQueue.PushFunction([this, &frame]()
		{
			vmaDestroyBuffer(allocator, frame.cameraBuffer.buffer, frame.cameraBuffer.allocation);
		});
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
	file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));

	// Now that the file is loaded into the buffer, we can close it
	file.close();

	// Create a new shader module, using the buffer we loaded
	VkShaderModuleCreateInfo createInfo = vkinit::ShaderModuleCreateInfo();

	// codeSize has to be in bytes, so multiply by the ints in the buffer by size of int to know the real size of the buffer
	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	// Check that the creation goes well
	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		return false;
	}

	*outShaderModule = shaderModule;
	return true;
}

void VulkanEngine::LoadMeshes()
{
	Mesh triangleMesh;
	triangleMesh.vertices.resize(3);

	triangleMesh.vertices[0].position = { 1.f, 1.f, 0.0f };
	triangleMesh.vertices[1].position = { -1.f, 1.f, 0.0f };
	triangleMesh.vertices[2].position = { 0.f, -1.f, 0.0f };

	//vertex colors, all green
	triangleMesh.vertices[0].color = { 0.f, 1.f, 0.0f }; //pure green
	triangleMesh.vertices[1].color = { 0.f, 1.f, 0.0f }; //pure green
	triangleMesh.vertices[2].color = { 0.f, 1.f, 0.0f }; //pure green

	// Load the monkey
	Mesh monkeyMesh;
	monkeyMesh.LoadFromObj("../../assets/monkey_smooth.obj");

	UploadMesh(triangleMesh);
	UploadMesh(monkeyMesh);

	meshes["monkey"] = monkeyMesh;
	meshes["triangle"] = triangleMesh;
}

void VulkanEngine::UploadMesh(Mesh& mesh)
{
	// Allocate vertex buffer
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

	// This is the total size, in bytes, of the buffer we are allocating
	bufferInfo.size = mesh.vertices.size() * sizeof(Vertex);

	// This buffer is going to be used as a Vertex buffer;
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;


	// Let the VMA library know that this data should be writable by CPU, but also readable by GPU
	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	// Allocate the buffer
	VK_CHECK(
		vmaCreateBuffer(allocator, &bufferInfo, &vmaAllocInfo, &mesh.vertexBuffer.buffer, &mesh.vertexBuffer.allocation,
		nullptr));

	// Add the destruction of the triangle mesh buffer to the deletion queue
	_mainDeletionQueue.PushFunction([this, mesh]()
	{
		vmaDestroyBuffer(allocator, mesh.vertexBuffer.buffer, mesh.vertexBuffer.allocation);
	});

	// Copy vertex data
	void* data;
	vmaMapMemory(allocator, mesh.vertexBuffer.allocation, &data);
	std::memcpy(data, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
	vmaUnmapMemory(allocator, mesh.vertexBuffer.allocation);
}

// ReSharper disable twice CppParameterMayBeConst
VkPipeline PipelineBuilder::BuildPipeline(VkDevice device, VkRenderPass pass) const
{
	// Make the viewport state from our stored viewport and scissor.
	// At the moment we won't support multiple viewports and scissors
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.pNext = nullptr;

	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	// Setup dummy color blending. We aren't using transparent objects yet
	// the blending is just "no blend", but we do write to the color attachment
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.pNext = nullptr;

	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;

	// Build the actual pipeline
	// We now use all of the info structs we have been writing into this one to create the pipeline
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

	pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	pipelineInfo.pDepthStencilState = &depthStencil;

	// It's easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS)
	{
		std::cerr << "Failed to create pipeline\n";
		return VK_NULL_HANDLE;
	}

	return newPipeline;
}
