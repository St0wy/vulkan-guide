#include "vk_mesh.h"

#include <tiny_obj_loader.h>
#include <iostream>

VertexInputDescription Vertex::GetVertexDescription()
{
	VertexInputDescription description;

	// We will have just 1 vertex buffer bindings, with a per-vertex rate
	VkVertexInputBindingDescription mainBinding{};
	mainBinding.binding = 0;
	mainBinding.stride = sizeof(Vertex);
	mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(mainBinding);

	// Position will be stored at Location 0
	VkVertexInputAttributeDescription positionAttribute{};
	positionAttribute.binding = 0;
	positionAttribute.location = 0;
	positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset = offsetof(Vertex, position);

	// Normal will be stored at Location 1
	VkVertexInputAttributeDescription normalAttribute{};
	normalAttribute.binding = 0;
	normalAttribute.location = 1;
	normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttribute.offset = offsetof(Vertex, normal);

	// Color will be stored at Location 2
	VkVertexInputAttributeDescription colorAttribute{};
	colorAttribute.binding = 0;
	colorAttribute.location = 2;
	colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttribute.offset = offsetof(Vertex, color);

	description.attributes.push_back(positionAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(colorAttribute);
	return description;
}

bool Mesh::LoadFromObj(const char* filename)
{
	// Attrib will contain the vertex array of the file
	tinyobj::attrib_t attrib;

	// Shapes contains the info for each separate object in the file
	std::vector<tinyobj::shape_t> shapes;

	// Materials contains the information about the material of each shape, but we won't use it
	std::vector<tinyobj::material_t> materials;

	// Error and warning about from the load function
	std::string warn;
	std::string err;

	// Load the OBJ file
	LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, nullptr);

	if (!warn.empty())
	{
		std::cout << "Warn: " << warn << "\n";
	}

	if (!err.empty())
	{
		std::cerr << err << std::endl;
		return false;
	}

	// Loop over shapes
	for (auto& [name, mesh, lines, points] : shapes)
	{
		// Loop over faces
		std::size_t indexOffset = 0;
		for (std::size_t f = 0; f < mesh.num_face_vertices.size(); f++)
		{
			// Hardcode loading to triangles
			constexpr std::size_t fv = 3;

			// Loop over vertices in the face
			for (std::size_t v = 0; v < fv; v++)
			{
				// Access to vertex
				auto [vertex_index, normal_index, texcoord_index] = mesh.indices[indexOffset + v];

				// Vertex position
				tinyobj::real_t vx = attrib.vertices[fv * vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[fv * vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[fv * vertex_index + 2];

				// Vertex normal
				tinyobj::real_t nx = attrib.normals[fv * normal_index + 0];
				tinyobj::real_t ny = attrib.normals[fv * normal_index + 1];
				tinyobj::real_t nz = attrib.normals[fv * normal_index + 2];

				// Copy it to our vertex
				Vertex newVertex{};
				newVertex.position.x = vx;
				newVertex.position.y = vy;
				newVertex.position.z = vz;
				newVertex.normal.x = nx;
				newVertex.normal.y = ny;
				newVertex.normal.z = nz;

				// We are setting the vertex color as the vertex normal. This is just for display purposes
				newVertex.color = newVertex.normal;

				vertices.push_back(newVertex);
			}

			indexOffset += fv;
		}
	}

	return true;
}
