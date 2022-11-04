﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_mem_alloc.h>

#include <vulkan/vulkan.h>

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
