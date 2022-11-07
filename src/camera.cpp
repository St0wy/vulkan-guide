#include "camera.h"

#include <algorithm>

Camera::Camera(const glm::vec3 position, const float aspectRatio)
	: position(position),
	aspectRatio(aspectRatio),
	_front(glm::vec3(0, 0.0f, -1.0f)),
	_up(glm::vec3(0, 1.0f, 0)),
	_right(1.0f, 0, 0),
	_pitch(0),
	_yaw(-glm::pi<float>() / 2.0f),
	_fov(glm::pi<float>() / 2.0f)
{}

void Camera::SetPitch(const float pitch)
{
	const float angle = std::clamp(pitch, glm::radians(-89.0f), glm::radians(89.0f));
	_pitch = angle;
	UpdateVectors();
}

void Camera::SetYaw(const float yaw)
{
	_yaw = yaw;
	UpdateVectors();
}

void Camera::SetFov(const float fov)
{
	const float angle = std::clamp(fov, glm::radians(1.0f), glm::radians(45.0f));
	_fov = angle;
}

glm::mat4 Camera::GetViewMatrix() const
{
	return lookAt(position, position + Front(), Up());
	//return glm::translate(glm::mat4{ 1.0f }, position);
}

glm::mat4 Camera::GetProjectionMatrix() const
{
	auto projection = glm::perspective(Fov(), aspectRatio, DEPTH_NEAR, DEPTH_FAR);
	projection[1][1] *= -1.0f;
	return projection;
}

void Camera::UpdateVectors()
{
	_front = glm::vec3{
		std::cos(Pitch()) * std::cos(Yaw()),
		std::sin(Pitch()),
		-std::cos(Pitch()) * std::sin(Yaw())
	};
	_front = normalize(_front);

	_up = glm::vec3(0, 1.0f, 0);
	_right = normalize(cross(_front, _up));
}
