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

	InitVulkan();
	InitSwapchain();
	InitCommands();
	InitDefaultRenderpass();
	InitFramebuffers();
	InitSyncStructures();
	InitPipelines();

	LoadMeshes();

	//everything went fine
	isInitialized = true;
}

void VulkanEngine::Cleanup()
{
	if (!isInitialized) return;

	// Make sure the gpu has stopped doing its things
	vkWaitForFences(device, 1, &renderFence, true, 1'000'000'000);

	_mainDeletionQueue.Flush();

	vkDestroySurfaceKHR(instance, surface, nullptr);

	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);

	SDL_DestroyWindow(window);
}

void VulkanEngine::Draw()
{
	// Wait until the GPU has finished rendering the last frame with a timeout of 1 second
	VK_CHECK(vkWaitForFences(device, 1, &renderFence, true, 1'000'000'000));
	VK_CHECK(vkResetFences(device, 1, &renderFence));

	// Request image from the swapchain, one second timeout
	uint32_t swapchainImageIndex;
	VK_CHECK(
		vkAcquireNextImageKHR(device, swapchain, 1'000'000'000, presentSemaphore, nullptr, &swapchainImageIndex));

	// Now that we are sure that the command finished executing, we can safely reset the command buffer to begin recording again
	VK_CHECK(vkResetCommandBuffer(mainCommandBuffer, 0));

	// ReSharper disable once CppLocalVariableMayBeConst
	VkCommandBuffer commandBuffer = mainCommandBuffer;

	// Begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan know that
	const VkCommandBufferBeginInfo beginInfo = vkinit::CommandBufferBeginInfo(
		VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	// Make a clear color form frame number. This will flash with a 120 * pi frame period
	VkClearValue clearValue{};
	const auto floatFrameNumber = static_cast<float>(frameNumber);
	const float flash = std::abs(std::sin(floatFrameNumber / 120.0f));
	clearValue.color = { {0.0f, 0.0f, flash, 1.0f} };

	// Start the main render pass.
	// We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo renderPassBeginInfo = vkinit::RenderPassBeginInfo(
		renderPass,
		windowExtent,
		framebuffers[swapchainImageIndex]);

	// Connect clear values
	renderPassBeginInfo.clearValueCount = 1;
	renderPassBeginInfo.pClearValues = &clearValue;

	vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);

	// Bind the mesh vertex buffer with offset 0
	constexpr VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &triangleMesh.vertexBuffer.buffer, &offset);

	// Make a model view matrix for rendering the object
	// Camera position
	glm::vec3 camPos = { 0.0f, 0.0f, -2.0f };

	glm::mat4 view = translate(glm::mat4(1.0f), camPos);
	glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1700.0f / 900.0f, 0.1f, 200.0f);
	projection[1][1] *= -1;
	float angle = glm::radians(static_cast<float>(frameNumber) * 0.4f);
	glm::mat4 model = rotate(glm::mat4{ 1.0f }, angle, glm::vec3(0, 1, 0));

	// Compute final mesh matrix
	glm::mat4 meshMatrix = projection * view * model;

	MeshPushConstants constants{};
	constants.renderMatrix = meshMatrix;

	// Upload the matrix to the GPU via push constants
	vkCmdPushConstants(commandBuffer, meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

	// We can now draw the mesh
	vkCmdDraw(commandBuffer, static_cast<uint32_t>(triangleMesh.vertices.size()), 1, 0, 0);

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
	submit.pWaitSemaphores = &presentSemaphore;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &renderSemaphore;

	// Submit command buffer to the queue and execute it
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submit, renderFence));

	// This will put the image we just rendered into the visible window
	// We want to wait on the _renderSemaphore for that,
	// as it's necessary that drawing command have finished before the image is displayed
	VkPresentInfoKHR presentInfo = vkinit::PresentInfo();

	presentInfo.pSwapchains = &swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &renderSemaphore;
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
}

void VulkanEngine::InitCommands()
{
	// Create a command pool for commands submitted to the graphics queue
	// We also want the pool to allow for resetting of individual command buffers
	const VkCommandPoolCreateInfo commandPoolInfo = vkinit::CommandPoolCreateInfo(
		graphicsQueueFamily,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool));

	// Allocate the default command buffer that we will use for rendering
	const VkCommandBufferAllocateInfo commandAllocateInfo = vkinit::CommandBufferAllocateInfo(commandPool, 1);

	VK_CHECK(vkAllocateCommandBuffers(device, &commandAllocateInfo, &mainCommandBuffer));

	_mainDeletionQueue.PushFunction([this]()
	{
		vkDestroyCommandPool(device, commandPool, nullptr);
	});
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
		framebufferInfo.pAttachments = &swapchainImageViews[i];
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

	VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &renderFence));

	_mainDeletionQueue.PushFunction([this]()
	{
		vkDestroyFence(device, renderFence, nullptr);
	});

	// For the semaphores we don't need any flags
	const VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::SemaphoreCreateInfo();

	VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentSemaphore));
	VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderSemaphore));

	_mainDeletionQueue.PushFunction([this]()
	{
		vkDestroySemaphore(device, presentSemaphore, nullptr);
		vkDestroySemaphore(device, renderSemaphore, nullptr);
	});
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

	VkPipelineLayoutCreateInfo meshPipelineLayoutInfo = vkinit::PipelineLayoutCreateInfo();

	// Setup push constants
	VkPushConstantRange pushConstant{};

	// This push constant starts at the begining
	pushConstant.offset = 0;

	// This push constant takes up the size of the MeshPushConstants struct
	pushConstant.size = sizeof(MeshPushConstants);

	// This push constant range is only accessible in the vertex shader
	pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	meshPipelineLayoutInfo.pPushConstantRanges = &pushConstant;
	meshPipelineLayoutInfo.pushConstantRangeCount = 1;

	VK_CHECK(vkCreatePipelineLayout(device, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout));

	pipelineBuilder.pipelineLayout = meshPipelineLayout;

	meshPipeline = pipelineBuilder.BuildPipeline(device, renderPass);

	vkDestroyShaderModule(device, meshVertexShader, nullptr);
	vkDestroyShaderModule(device, triangleFragShader, nullptr);

	_mainDeletionQueue.PushFunction([this]()
	{
		// Destroy the pipeline we have created
		vkDestroyPipeline(device, meshPipeline, nullptr);

		vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
	});
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
	triangleMesh.vertices.resize(3);

	triangleMesh.vertices[0].position = { 1.f, 1.f, 0.0f };
	triangleMesh.vertices[1].position = { -1.f, 1.f, 0.0f };
	triangleMesh.vertices[2].position = { 0.f, -1.f, 0.0f };

	//vertex colors, all green
	triangleMesh.vertices[0].color = { 0.f, 1.f, 0.0f }; //pure green
	triangleMesh.vertices[1].color = { 0.f, 1.f, 0.0f }; //pure green
	triangleMesh.vertices[2].color = { 0.f, 1.f, 0.0f }; //pure green

	UploadMesh(triangleMesh);
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

	// It's easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS)
	{
		std::cerr << "Failed to create pipeline\n";
		return VK_NULL_HANDLE;
	}

	return newPipeline;
}
