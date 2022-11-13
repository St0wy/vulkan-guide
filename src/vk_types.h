// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

struct Mesh;

struct AllocatedBuffer
{
	VkBuffer buffer;
	VmaAllocation allocation;
};

struct AllocatedImage
{
	VkImage image;
	VmaAllocation allocation;
};

struct Material
{
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject
{
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

struct GpuCameraData
{
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewProjection;
};

struct FrameData
{
	VkSemaphore presentSemaphore, renderSemaphore;
	VkFence renderFence;

	VkCommandPool commandPool;
	VkCommandBuffer mainCommandBuffer;

	AllocatedBuffer cameraBuffer;
	VkDescriptorSet globalDescriptor;

	AllocatedBuffer objectBuffer;
	VkDescriptorSet objectDescriptor;
};

struct GpuSceneData
{
    // w is for exponent
	glm::vec4 fogColor;
	// x for min, y for max, zw unused
	glm::vec4 fogDistances;
	glm::vec4 ambientColor;
	// w for sun power
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
};

struct GpuObjectData
{
	glm::mat4 modelMatrix;
};