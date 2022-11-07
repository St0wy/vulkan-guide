#pragma once

#include "vk_types.h"
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

class Camera
{
public:
	Camera(glm::vec3 position, float aspectRatio);

	glm::vec3 position;
	float aspectRatio;

	[[nodiscard]] glm::vec3 Front() const { return _front; }
	[[nodiscard]] glm::vec3 Up() const { return _up; }
	[[nodiscard]] glm::vec3 Right() const { return _right; }

	[[nodiscard]] float Pitch() const { return _pitch; }
	void SetPitch(float pitch);

	[[nodiscard]] float Yaw() const { return _yaw; }
	void SetYaw(float yaw);

	[[nodiscard]] float Fov() const { return _fov; }
	void SetFov(float fov);

	[[nodiscard]] glm::mat4 GetViewMatrix() const;

	[[nodiscard]] glm::mat4 GetProjectionMatrix() const;
private:
	static constexpr float DEPTH_NEAR = 0.01f;
	static constexpr float DEPTH_FAR = 1000.0f;

	glm::vec3 _front;
	glm::vec3 _up;
	glm::vec3 _right;

	float _pitch;
	float _yaw;
	float _fov = glm::radians(90.0f);

	void UpdateVectors();
};
