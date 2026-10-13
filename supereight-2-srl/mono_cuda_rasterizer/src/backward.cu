/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

// Faster-GS portions, modified for GOFusion:
// Copyright 2026 The Faster-GS Authors.
// These portions are licensed under Apache-2.0; see
// supereight-2-srl/fastergs_cuda_rasterizer/LICENSE.

#include "mono_backward.h"
#include "mono_auxiliary.h"
#include "mono_math.h"
#include "gs/surfel_geometry.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void mono_computeColorFromSH_bwd(int idx,
	int deg,
	int max_coeffs,
	const glm::vec3* means,
	glm::vec3 campos,
	const float* shs_dc,
	const float* shs_rest,
	const bool* clamped,
	const glm::vec3* dL_dcolor,
	glm::vec3* dL_dmeans,
	glm::vec3* dL_dsh_dc,
	glm::vec3* dL_dsh_rest,
	float *dL_dtau)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	const int rest_coeffs = max_coeffs - 1;
	const glm::vec3* sh_rest =
		(rest_coeffs > 0 && shs_rest != nullptr)
		? (reinterpret_cast<const glm::vec3*>(shs_rest) + idx * rest_coeffs)
		: nullptr;
	glm::vec3* dL_dc = dL_dsh_dc + idx;
	glm::vec3* dL_rest =
		(rest_coeffs > 0 && dL_dsh_rest != nullptr)
		? (dL_dsh_rest + idx * rest_coeffs)
		: nullptr;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = MONO_SH_C0;
	dL_dc[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -MONO_SH_C1 * y;
		float dRGBdsh2 = MONO_SH_C1 * z;
		float dRGBdsh3 = -MONO_SH_C1 * x;
		dL_rest[0] = dRGBdsh1 * dL_dRGB;
		dL_rest[1] = dRGBdsh2 * dL_dRGB;
		dL_rest[2] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -MONO_SH_C1 * sh_rest[2];
		dRGBdy = -MONO_SH_C1 * sh_rest[0];
		dRGBdz = MONO_SH_C1 * sh_rest[1];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = MONO_SH_C2[0] * xy;
			float dRGBdsh5 = MONO_SH_C2[1] * yz;
			float dRGBdsh6 = MONO_SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = MONO_SH_C2[3] * xz;
			float dRGBdsh8 = MONO_SH_C2[4] * (xx - yy);
			dL_rest[3] = dRGBdsh4 * dL_dRGB;
			dL_rest[4] = dRGBdsh5 * dL_dRGB;
			dL_rest[5] = dRGBdsh6 * dL_dRGB;
			dL_rest[6] = dRGBdsh7 * dL_dRGB;
			dL_rest[7] = dRGBdsh8 * dL_dRGB;

			dRGBdx += MONO_SH_C2[0] * y * sh_rest[3] + MONO_SH_C2[2] * 2.f * -x * sh_rest[5] + MONO_SH_C2[3] * z * sh_rest[6] + MONO_SH_C2[4] * 2.f * x * sh_rest[7];
			dRGBdy += MONO_SH_C2[0] * x * sh_rest[3] + MONO_SH_C2[1] * z * sh_rest[4] + MONO_SH_C2[2] * 2.f * -y * sh_rest[5] + MONO_SH_C2[4] * 2.f * -y * sh_rest[7];
			dRGBdz += MONO_SH_C2[1] * y * sh_rest[4] + MONO_SH_C2[2] * 2.f * 2.f * z * sh_rest[5] + MONO_SH_C2[3] * x * sh_rest[6];

			if (deg > 2)
			{
				float dRGBdsh9 = MONO_SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = MONO_SH_C3[1] * xy * z;
				float dRGBdsh11 = MONO_SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = MONO_SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = MONO_SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = MONO_SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = MONO_SH_C3[6] * x * (xx - 3.f * yy);
				dL_rest[8] = dRGBdsh9 * dL_dRGB;
				dL_rest[9] = dRGBdsh10 * dL_dRGB;
				dL_rest[10] = dRGBdsh11 * dL_dRGB;
				dL_rest[11] = dRGBdsh12 * dL_dRGB;
				dL_rest[12] = dRGBdsh13 * dL_dRGB;
				dL_rest[13] = dRGBdsh14 * dL_dRGB;
				dL_rest[14] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					MONO_SH_C3[0] * sh_rest[8] * 3.f * 2.f * xy +
					MONO_SH_C3[1] * sh_rest[9] * yz +
					MONO_SH_C3[2] * sh_rest[10] * -2.f * xy +
					MONO_SH_C3[3] * sh_rest[11] * -3.f * 2.f * xz +
					MONO_SH_C3[4] * sh_rest[12] * (-3.f * xx + 4.f * zz - yy) +
					MONO_SH_C3[5] * sh_rest[13] * 2.f * xz +
					MONO_SH_C3[6] * sh_rest[14] * 3.f * (xx - yy));

				dRGBdy += (
					MONO_SH_C3[0] * sh_rest[8] * 3.f * (xx - yy) +
					MONO_SH_C3[1] * sh_rest[9] * xz +
					MONO_SH_C3[2] * sh_rest[10] * (-3.f * yy + 4.f * zz - xx) +
					MONO_SH_C3[3] * sh_rest[11] * -3.f * 2.f * yz +
					MONO_SH_C3[4] * sh_rest[12] * -2.f * xy +
					MONO_SH_C3[5] * sh_rest[13] * -2.f * yz +
					MONO_SH_C3[6] * sh_rest[14] * -3.f * 2.f * xy);

				dRGBdz += (
					MONO_SH_C3[1] * sh_rest[9] * xy +
					MONO_SH_C3[2] * sh_rest[10] * 4.f * 2.f * yz +
					MONO_SH_C3[3] * sh_rest[11] * 3.f * (2.f * zz - xx - yy) +
					MONO_SH_C3[4] * sh_rest[12] * 4.f * 2.f * xz +
					MONO_SH_C3[5] * sh_rest[13] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = mono_dnormvdv_aux(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);

	if (dL_dtau != nullptr) {
		dL_dtau[6 * idx + 0] += -dL_dmean.x;
		dL_dtau[6 * idx + 1] += -dL_dmean.y;
		dL_dtau[6 * idx + 2] += -dL_dmean.z;
	}

}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void mono_computeCov2DCUDA_bwd(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float *dL_dtau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = mono_transformPoint4x3_aux(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	SE3 T_CW(view_matrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t_ = T_CW.t();
	mat33 dpC_drho = mat33::identity();
	mat33 dpC_dtheta = -mat33::skew_symmetric(t);
	float dL_dt[6];
	for (int i = 0; i < 3; i++) {
		float3 c_rho = dpC_drho.cols[i];
		float3 c_theta = dpC_dtheta.cols[i];
		dL_dt[i] = dL_dtx * c_rho.x + dL_dty * c_rho.y + dL_dtz * c_rho.z;
		dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;
	}
	if (dL_dtau != nullptr) {
		for (int i = 0; i < 6; i++) {
			dL_dtau[6 * idx + i] += dL_dt[i];
		}
	}

	// Account for transformation of mean to t
	// t = mono_transformPoint4x3_aux(mean, view_matrix);
	float3 dL_dmean = mono_transformVec4x3Transpose_aux({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in MONO_BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;

	float dL_dW00 = J[0][0] * dL_dT00;
	float dL_dW01 = J[0][0] * dL_dT01;
	float dL_dW02 = J[0][0] * dL_dT02;
	float dL_dW10 = J[1][1] * dL_dT10;
	float dL_dW11 = J[1][1] * dL_dT11;
	float dL_dW12 = J[1][1] * dL_dT12;
	float dL_dW20 = J[0][2] * dL_dT00 + J[1][2] * dL_dT10;
	float dL_dW21 = J[0][2] * dL_dT01 + J[1][2] * dL_dT11;
	float dL_dW22 = J[0][2] * dL_dT02 + J[1][2] * dL_dT12;

	float3 c1 = R.cols[0];
	float3 c2 = R.cols[1];
	float3 c3 = R.cols[2];

	float dL_dW_data[9];
	dL_dW_data[0] = dL_dW00;
	dL_dW_data[3] = dL_dW01;
	dL_dW_data[6] = dL_dW02;
	dL_dW_data[1] = dL_dW10;
	dL_dW_data[4] = dL_dW11;
	dL_dW_data[7] = dL_dW12;
	dL_dW_data[2] = dL_dW20;
	dL_dW_data[5] = dL_dW21;
	dL_dW_data[8] = dL_dW22;

	mat33 dL_dW(dL_dW_data);
	float3 dL_dWc1 = dL_dW.cols[0];
	float3 dL_dWc2 = dL_dW.cols[1];
	float3 dL_dWc3 = dL_dW.cols[2];

	mat33 n_W1_x = -mat33::skew_symmetric(c1);
	mat33 n_W2_x = -mat33::skew_symmetric(c2);
	mat33 n_W3_x = -mat33::skew_symmetric(c3);

	float3 dL_dtheta = {};
	dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0]) +
				dot(dL_dWc3, n_W3_x.cols[0]);
	dL_dtheta.y = dot(dL_dWc1, n_W1_x.cols[1]) + dot(dL_dWc2, n_W2_x.cols[1]) +
				dot(dL_dWc3, n_W3_x.cols[1]);
	dL_dtheta.z = dot(dL_dWc1, n_W1_x.cols[2]) + dot(dL_dWc2, n_W2_x.cols[2]) +
				dot(dL_dWc3, n_W3_x.cols[2]);

	if (dL_dtau != nullptr) {
		dL_dtau[6 * idx + 3] += dL_dtheta.x;
		dL_dtau[6 * idx + 4] += dL_dtheta.y;
		dL_dtau[6 * idx + 5] += dL_dtheta.z;
	}


}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void mono_computeCov3D_bwd(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	const float q_norm2 = rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w;
	float4* dL_drot = dL_drots ? (float4*)(dL_drots + idx) : nullptr;
	if (q_norm2 <= MONO_MIN_QUAT_NORM2) {
		return;
	}
	const float inv_q_norm = rsqrtf(q_norm2);
	glm::vec4 q = rot * inv_q_norm;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	if (dL_dscales != nullptr) {
		glm::vec3* dL_dscale = dL_dscales + idx;
		dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
		dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
		dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);
	}

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	if (dL_drot != nullptr) {
		*dL_drot += mono_dnormvdv_aux(
			float4{ rot.x, rot.y, rot.z, rot.w },
			float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
	}
}

namespace {
constexpr int kMonoMaxShCoeffs = 16;
constexpr int kMonoMaxShRestCoeffs = kMonoMaxShCoeffs - 1;

__device__ inline void monoAccumulateSquareLocalScalar(float* accum, float value)
{
	if (accum == nullptr) {
		return;
	}
	*accum += value * value;
}

__device__ inline void monoAccumulateSquareLocalVec3(float* accum, const float3& value)
{
	if (accum == nullptr) {
		return;
	}
	accum[0] += value.x * value.x;
	accum[1] += value.y * value.y;
	accum[2] += value.z * value.z;
}


__device__ inline void monoAtomicAddScalar(float* address, float value)
{
	if (address == nullptr) {
		return;
	}
	atomicAdd(address, value);
}

__device__ inline void monoAtomicAddArray(float* address, const float* values, int count)
{
	if (address == nullptr) {
		return;
	}
	for (int i = 0; i < count; ++i) {
		atomicAdd(address + i, values[i]);
	}
}


__device__ inline void mono_computeCov2D_mean_only_camera_bwd(
	const float3& mean,
	const float* cov3D,
	const float h_x,
	const float h_y,
	const float tan_fovx,
	const float tan_fovy,
	const float* view_matrix,
	const float3& dL_dconic,
	float3* dL_dp_camera)
{
	float3 t = mono_transformPoint4x3_aux(mean, view_matrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	const float x_grad_mul = txtz < -limx || txtz > limx ? 0.0f : 1.0f;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0.0f : 1.0f;

	glm::mat3 J = glm::mat3(h_x / t.z,
	                        0.0f,
	                        -(h_x * t.x) / (t.z * t.z),
	                        0.0f,
	                        h_y / t.z,
	                        -(h_y * t.y) / (t.z * t.z),
	                        0.0f,
	                        0.0f,
	                        0.0f);

	glm::mat3 W = glm::mat3(view_matrix[0],
	                        view_matrix[4],
	                        view_matrix[8],
	                        view_matrix[1],
	                        view_matrix[5],
	                        view_matrix[9],
	                        view_matrix[2],
	                        view_matrix[6],
	                        view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(cov3D[0],
	                          cov3D[1],
	                          cov3D[2],
	                          cov3D[1],
	                          cov3D[3],
	                          cov3D[4],
	                          cov3D[2],
	                          cov3D[4],
	                          cov3D[5]);

	glm::mat3 T = W * J;
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	float a = cov2D[0][0] + 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] + 0.3f;

	float dL_da = 0.0f;
	float dL_db = 0.0f;
	float dL_dc = 0.0f;
	const float denom = a * c - b * b;
	const float denom2inv = 1.0f / ((denom * denom) + 1.0e-7f);

	if (denom2inv != 0.0f) {
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2.0f * b * c * dL_dconic.y
		                     + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2.0f * a * b * dL_dconic.y
		                     + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2.0f
		        * (b * c * dL_dconic.x - (denom + 2.0f * b * b) * dL_dconic.y
		           + a * b * dL_dconic.z);
	}

	float dL_dT00 = 2.0f * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da
	                + (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2.0f * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da
	                + (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2.0f * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da
	                + (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2.0f * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc
	                + (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2.0f * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc
	                + (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2.0f * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc
	                + (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	const float tz = 1.0f / t.z;
	const float tz2 = tz * tz;
	const float tz3 = tz2 * tz;

	dL_dp_camera->x += x_grad_mul * -h_x * tz2 * dL_dJ02;
	dL_dp_camera->y += y_grad_mul * -h_y * tz2 * dL_dJ12;
	dL_dp_camera->z += -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11
	                   + (2.0f * h_x * t.x) * tz3 * dL_dJ02
	                   + (2.0f * h_y * t.y) * tz3 * dL_dJ12;
}

__device__ inline void mono_projectMean2D_camera_bwd(
	const float3& mean,
	const float* view_matrix,
	const float focal_x,
	const float focal_y,
	const float2& dL_dmean2D,
	float3* dL_dp_camera)
{
	const float3 t = mono_transformPoint4x3_aux(mean, view_matrix);
	const float tz = 1.0f / (t.z + 1.0e-7f);
	const float tz2 = tz * tz;

	dL_dp_camera->x += focal_x * tz * dL_dmean2D.x;
	dL_dp_camera->y += focal_y * tz * dL_dmean2D.y;
	dL_dp_camera->z += -(focal_x * t.x) * tz2 * dL_dmean2D.x
	                   -(focal_y * t.y) * tz2 * dL_dmean2D.y;
}

__device__ inline void mono_accumulate_point_fisher_camera_space(
	const uint32_t primitive_idx,
	const int P,
	const float3* means3D,
	const float* cov3Ds,
	const float* viewmatrix,
	const float focal_x,
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const float2& cur_dL_dmean2D,
	const float3& cur_dL_dconic2D,
	const float cur_dL_dopacity,
	float* fisher_point_dL_dmean3D_accum,
	float* fisher_point_dL_dopacity_accum)
{
	if (primitive_idx >= static_cast<uint32_t>(P)) {
		return;
	}

	float3 cur_dL_dmeans_camera = make_float3(0.0f, 0.0f, 0.0f);
	mono_computeCov2D_mean_only_camera_bwd(
		means3D[primitive_idx],
		cov3Ds + 6 * primitive_idx,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		cur_dL_dconic2D,
		&cur_dL_dmeans_camera);
	mono_projectMean2D_camera_bwd(
		means3D[primitive_idx],
		viewmatrix,
		focal_x,
		focal_y,
		cur_dL_dmean2D,
		&cur_dL_dmeans_camera);

	monoAccumulateSquareLocalVec3(
		fisher_point_dL_dmean3D_accum,
		cur_dL_dmeans_camera);
	monoAccumulateSquareLocalScalar(
		fisher_point_dL_dopacity_accum,
		cur_dL_dopacity);
}

__device__ inline void mono_accumulate_fisher_diag_pixel(
	const uint32_t primitive_idx,
	const int P,
	const float3* means3D,
	const float* cov3Ds,
	const float* viewmatrix,
	const float focal_x,
	const float focal_y,
	const float ddelx_dx,
	const float ddely_dy,
	const float tan_fovx,
	const float tan_fovy,
	const float3& grad_color_pixel,
	const float3& conic,
	const float opacity,
	const float3& color,
	const float transmittance,
	const float grad_alpha_common,
	const float2& delta,
	const float gaussian,
	const float alpha,
	const float blending_weight,
	const float one_minus_alpha_rcp,
	const float3& color_pixel_after,
	float* fisher_dL_dmean3D_accum,
	float* fisher_dL_dopacity_accum)
{
	if (primitive_idx >= static_cast<uint32_t>(P)) {
		return;
	}

	const float dL_dalpha_from_color = dot(
		transmittance * color - color_pixel_after * one_minus_alpha_rcp,
		grad_color_pixel);
	const float dL_dalpha = dL_dalpha_from_color + grad_alpha_common * one_minus_alpha_rcp;
	const float gaussian_grad_helper = -alpha * dL_dalpha;
	const float2 cur_dL_dmean2D = make_float2(
		gaussian_grad_helper * (conic.x * delta.x + conic.y * delta.y) * ddelx_dx,
		gaussian_grad_helper * (conic.y * delta.x + conic.z * delta.y) * ddely_dy);
	const float3 cur_dL_dconic2D = make_float3(
		0.5f * gaussian_grad_helper * delta.x * delta.x,
		0.5f * gaussian_grad_helper * delta.x * delta.y,
		0.5f * gaussian_grad_helper * delta.y * delta.y);
	const float cur_dL_dopacity = gaussian * dL_dalpha;
	mono_accumulate_point_fisher_camera_space(
		primitive_idx,
		P,
		means3D,
		cov3Ds,
		viewmatrix,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		cur_dL_dmean2D,
		cur_dL_dconic2D,
		cur_dL_dopacity,
		fisher_dL_dmean3D_accum,
		fisher_dL_dopacity_accum);
}
} // namespace

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void mono_preprocessCUDA_bwd(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs_dc,
	const float* shs_rest,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float *viewmatrix,
	const float* proj,
	const float *proj_raw,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float *dL_ddepth,
	const float4* dL_dplanes,
	float* dL_dcov3D,
	float* dL_dsh_dc,
	float* dL_dsh_rest,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float *dL_dtau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = mono_transformPoint4x4_aux(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	float a = proj_raw[0];
	float b = proj_raw[5];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t = T_CW.t();
	float3 p_C = T_CW * m;
	mat33 dp_C_d_rho = mat33::identity();
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	float dL_dt[6];
	for (int i = 0; i < 6; i++) {
		dL_dt[i] = dL_dmean2D[idx].x * dmean2D_dtau[i].x + dL_dmean2D[idx].y * dmean2D_dtau[i].y;
	}
	if (dL_dtau != nullptr) {
		for (int i = 0; i < 6; i++) {
			dL_dtau[6 * idx + i] += dL_dt[i];
		}
	}

	// Compute gradient update due to computing depths
	// p_orig = m
	// p_view = mono_transformPoint4x3_aux(p_orig, viewmatrix);
	// depth = p_view.z;
	float dL_dpCz = dL_ddepth[idx];
	dL_dmeans[idx].x += dL_dpCz * viewmatrix[2];
	dL_dmeans[idx].y += dL_dpCz * viewmatrix[6];
	dL_dmeans[idx].z += dL_dpCz * viewmatrix[10];

	if (dL_dtau != nullptr) {
		for (int i = 0; i < 3; i++) {
			float3 c_rho = dp_C_d_rho.cols[i];
			float3 c_theta = dp_C_d_theta.cols[i];
			dL_dtau[6 * idx + i] += dL_dpCz * c_rho.z;
			dL_dtau[6 * idx + i + 3] += dL_dpCz * c_theta.z;
		}
	}

	// Propagate the exact camera-space surfel plane back to the Gaussian
	// center, quaternion and (when requested) camera pose.
	if (dL_dplanes != nullptr && rotations != nullptr) {
		const float4 plane_grad = dL_dplanes[idx];
		const float3 normal_world = gs::surfel::quaternionNormal(
			make_float4(rotations[idx].x, rotations[idx].y,
				rotations[idx].z, rotations[idx].w));
		float3 normal_camera_raw = R * normal_world;
		const float orientation = dot(normal_camera_raw, p_C) > 0.0f ? -1.0f : 1.0f;
		const float3 normal_camera = orientation * normal_camera_raw;

		const float3 dL_dp_C = plane_grad.w * normal_camera;
		const float3 dL_dn_C = orientation * (make_float3(
			plane_grad.x, plane_grad.y, plane_grad.z) + plane_grad.w * p_C);
		const float3 dL_dm_W = RT * dL_dp_C;
		dL_dmeans[idx].x += dL_dm_W.x;
		dL_dmeans[idx].y += dL_dm_W.y;
		dL_dmeans[idx].z += dL_dm_W.z;

		const float3 dL_dn_W = RT * dL_dn_C;
		const float4 dL_dq = gs::surfel::quaternionNormalBackward(
			make_float4(rotations[idx].x, rotations[idx].y,
				rotations[idx].z, rotations[idx].w), dL_dn_W);
		if (dL_drot != nullptr) {
			dL_drot[idx].x += dL_dq.x;
			dL_drot[idx].y += dL_dq.y;
			dL_drot[idx].z += dL_dq.z;
			dL_drot[idx].w += dL_dq.w;
		}

		if (dL_dtau != nullptr) {
			const float3 dL_dtheta =
				dp_C_d_theta.transpose() * dL_dp_C
				+ (-mat33::skew_symmetric(normal_camera_raw)).transpose() * dL_dn_C;
			dL_dtau[6 * idx + 0] += dL_dp_C.x;
			dL_dtau[6 * idx + 1] += dL_dp_C.y;
			dL_dtau[6 * idx + 2] += dL_dp_C.z;
			dL_dtau[6 * idx + 3] += dL_dtheta.x;
			dL_dtau[6 * idx + 4] += dL_dtheta.y;
			dL_dtau[6 * idx + 5] += dL_dtheta.z;
		}
	}

	// Compute gradient updates due to computing colors from SHs
	if (M > 0 && shs_dc && dL_dsh_dc)
		mono_computeColorFromSH_bwd(idx,
			D,
			M,
			(glm::vec3*)means,
			*campos,
			shs_dc,
			shs_rest,
			clamped,
			(glm::vec3*)dL_dcolor,
			(glm::vec3*)dL_dmeans,
			(glm::vec3*)dL_dsh_dc,
			(glm::vec3*)dL_dsh_rest,
			dL_dtau);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		mono_computeCov3D_bwd(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward pass with one warp per 32-Gaussian bucket.
template <uint32_t C, bool CAPTURE_FISHER_DIAG, bool RENDER_GEOMETRY>
__global__ void mono_renderCUDA_bucket_bwd(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ tile_bucket_offsets,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float4* __restrict__ planes,
	const float* __restrict__ rendered_color,
	const float* __restrict__ rendered_depth,
	const float* __restrict__ rendered_opacity,
	const float* __restrict__ rendered_normal,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ tile_max_n_processed,
	const uint32_t* __restrict__ bucket_tile_index,
	const float4* __restrict__ bucket_color_transmittance,
	const float* __restrict__ bucket_depth_accum,
	const float4* __restrict__ bucket_geometry_accum,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_depth,
	const float* __restrict__ dL_dpixels_opacity,
	const float* __restrict__ dL_dpixels_normal,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	float4* __restrict__ dL_dplanes,
	float* __restrict__ fisher_dL_dmean3D,
	float* __restrict__ fisher_dL_dopacity,
	int P,
	int D,
	int M,
	const float3* __restrict__ means3D,
	const float* __restrict__ shs_dc,
	const float* __restrict__ shs_rest,
	const bool* __restrict__ clamped,
	const glm::vec3* __restrict__ scales,
	const glm::vec4* __restrict__ rotations,
	const float scale_modifier,
	const float* __restrict__ cov3Ds,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	const float focal_x,
	const float focal_y,
	const float center_x,
	const float center_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* __restrict__ campos)
{
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<32>(block);
	const uint32_t bucket_idx = block.group_index().x;
	const uint32_t lane_idx = warp.thread_rank();

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint32_t tile_idx = bucket_tile_index[bucket_idx];
	const uint2 tile_range = ranges[tile_idx];
	const int tile_n_primitives = tile_range.y - tile_range.x;
	const uint32_t tile_first_bucket_offset = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];
	const int tile_bucket_idx = static_cast<int>(bucket_idx - tile_first_bucket_offset);
	if (tile_bucket_idx * 32 >= static_cast<int>(tile_max_n_processed[tile_idx])) {
		return;
	}

	const int tile_primitive_idx = tile_bucket_idx * 32 + static_cast<int>(lane_idx);
	const int instance_idx = static_cast<int>(tile_range.x) + tile_primitive_idx;
	const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

	uint32_t primitive_idx = 0;
	float2 mean2d = {0.0f, 0.0f};
	float3 conic = {0.0f, 0.0f, 0.0f};
	float opacity = 0.0f;
	float3 color = {0.0f, 0.0f, 0.0f};
	float depth = 0.0f;
	float4 plane = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (valid_primitive) {
		primitive_idx = point_list[instance_idx];
		mean2d = points_xy_image[primitive_idx];
		const float4 conic_op = conic_opacity[primitive_idx];
		conic = make_float3(conic_op);
		opacity = conic_op.w;
		color = make_float3(
			colors[primitive_idx * C + 0],
			colors[primitive_idx * C + 1],
			colors[primitive_idx * C + 2]);
		depth = depths[primitive_idx];
		if constexpr (RENDER_GEOMETRY) plane = planes[primitive_idx];
	}

	const float3 background = {bg_color[0], bg_color[1], bg_color[2]};
	const uint32_t n_pixels = static_cast<uint32_t>(W * H);
	const float ddelx_dx = 0.5f * static_cast<float>(W);
	const float ddely_dy = 0.5f * static_cast<float>(H);

	float2 dL_dmean2d_accum = {0.0f, 0.0f};
	float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
	float dL_dopacity_accum = 0.0f;
	float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};
	float dL_ddepth_accum = 0.0f;
	float4 dL_dplane_accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	
	float fisher_dL_dmean3D_accum[3] = {0.0f, 0.0f, 0.0f};
	float fisher_dL_dopacity_accum = 0.0f;

	const uint2 tile_coords = {tile_idx % horizontal_blocks, tile_idx / horizontal_blocks};
	const uint2 start_pixel_coords = {tile_coords.x * BLOCK_X, tile_coords.y * BLOCK_Y};

	uint32_t last_contributor = 0;
	float3 color_pixel_after = {0.0f, 0.0f, 0.0f};
	float depth_pixel_after = 0.0f;
	float3 normal_pixel_after = make_float3(0.0f, 0.0f, 0.0f);
	float transmittance = 0.0f;
	float final_trans = 0.0f;
	float3 grad_color_pixel = {0.0f, 0.0f, 0.0f};
	float grad_depth_pixel = 0.0f;
	float3 grad_normal_pixel = make_float3(0.0f, 0.0f, 0.0f);
	float grad_alpha_common = 0.0f;

	bucket_color_transmittance += bucket_idx * BLOCK_SIZE;
	bucket_depth_accum += bucket_idx * BLOCK_SIZE;
	if constexpr (RENDER_GEOMETRY) bucket_geometry_accum += bucket_idx * BLOCK_SIZE;
	__shared__ uint32_t collected_last_contributor[32];
	__shared__ float4 collected_color_pixel_after_transmittance[32];
	__shared__ float collected_depth_pixel_after[32];
	__shared__ float4 collected_geometry_pixel_after[RENDER_GEOMETRY ? 32 : 1];
	__shared__ float4 collected_grad_info_pixel[32];
	__shared__ float collected_grad_depth_pixel[32];
	__shared__ float4 collected_grad_geometry_pixel[RENDER_GEOMETRY ? 32 : 1];
	__shared__ float collected_final_trans[32];

	#pragma unroll
	for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
		if (i % 32 == 0) {
			const uint32_t local_idx = static_cast<uint32_t>(i) + lane_idx;
			if (local_idx < BLOCK_SIZE) {
				const float4 color_trans = bucket_color_transmittance[local_idx];
				const float depth_before_bucket = bucket_depth_accum[local_idx];
					float4 geometry_before_bucket =
						make_float4(depth_before_bucket, 0.0f, 0.0f, 0.0f);
					if constexpr (RENDER_GEOMETRY) {
						geometry_before_bucket = bucket_geometry_accum[local_idx];
					}
				const uint2 pixel_coords = {
					start_pixel_coords.x + (local_idx % BLOCK_X),
					start_pixel_coords.y + (local_idx / BLOCK_X)
				};
				const bool valid_pixel = pixel_coords.x < static_cast<uint32_t>(W)
					&& pixel_coords.y < static_cast<uint32_t>(H);
				const uint32_t pixel_idx = static_cast<uint32_t>(W) * pixel_coords.y + pixel_coords.x;

				float3 color_pixel_w_bg = {0.0f, 0.0f, 0.0f};
				float depth_pixel_final = 0.0f;
				float3 grad_pixel = {0.0f, 0.0f, 0.0f};
				float grad_pixel_depth = 0.0f;
				float3 normal_pixel_final = make_float3(0.0f, 0.0f, 0.0f);
				float3 grad_pixel_normal = make_float3(0.0f, 0.0f, 0.0f);
				float grad_pixel_opacity = 0.0f;
				float final_trans = 1.0f;
				uint32_t contrib = 0;
				if (valid_pixel) {
					color_pixel_w_bg = make_float3(
						rendered_color[pixel_idx],
						rendered_color[n_pixels + pixel_idx],
						rendered_color[2 * n_pixels + pixel_idx]);
					depth_pixel_final = rendered_depth[pixel_idx];
					grad_pixel = make_float3(
						dL_dpixels[pixel_idx],
						dL_dpixels[n_pixels + pixel_idx],
						dL_dpixels[2 * n_pixels + pixel_idx]);
					grad_pixel_depth = dL_dpixels_depth[pixel_idx];
						if constexpr (RENDER_GEOMETRY) {
						normal_pixel_final = make_float3(
							rendered_normal[pixel_idx],
							rendered_normal[n_pixels + pixel_idx],
							rendered_normal[2 * n_pixels + pixel_idx]);
						grad_pixel_normal = make_float3(
							dL_dpixels_normal[pixel_idx],
							dL_dpixels_normal[n_pixels + pixel_idx],
							dL_dpixels_normal[2 * n_pixels + pixel_idx]);
						grad_pixel_opacity = dL_dpixels_opacity[pixel_idx];
					}
					final_trans = final_Ts[pixel_idx];
					contrib = n_contrib[pixel_idx];
				}

				collected_color_pixel_after_transmittance[lane_idx] = make_float4(
					color_pixel_w_bg - final_trans * background - make_float3(color_trans),
					color_trans.w);
				collected_depth_pixel_after[lane_idx] = depth_pixel_final - depth_before_bucket;
					if constexpr (RENDER_GEOMETRY) {
						collected_geometry_pixel_after[lane_idx] = make_float4(
							depth_pixel_final - geometry_before_bucket.x,
							normal_pixel_final.x - geometry_before_bucket.y,
							normal_pixel_final.y - geometry_before_bucket.z,
							normal_pixel_final.z - geometry_before_bucket.w);
					}
				collected_grad_info_pixel[lane_idx] = make_float4(
					grad_pixel,
					final_trans * -dot(grad_pixel, background));
				collected_grad_depth_pixel[lane_idx] = grad_pixel_depth;
					if constexpr (RENDER_GEOMETRY) {
						collected_grad_geometry_pixel[lane_idx] = make_float4(
							grad_pixel_opacity, grad_pixel_normal.x,
							grad_pixel_normal.y, grad_pixel_normal.z);
					}
				collected_final_trans[lane_idx] = final_trans;
				collected_last_contributor[lane_idx] = contrib;
			}
			warp.sync();
		}

			if (i > 0) {
			last_contributor = warp.shfl_up(last_contributor, 1);
			color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
			color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
			color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
			depth_pixel_after = warp.shfl_up(depth_pixel_after, 1);
				if constexpr (RENDER_GEOMETRY) {
					normal_pixel_after.x = warp.shfl_up(normal_pixel_after.x, 1);
					normal_pixel_after.y = warp.shfl_up(normal_pixel_after.y, 1);
					normal_pixel_after.z = warp.shfl_up(normal_pixel_after.z, 1);
				}
			transmittance = warp.shfl_up(transmittance, 1);
			final_trans = warp.shfl_up(final_trans, 1);
			grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
			grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
			grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
			grad_depth_pixel = warp.shfl_up(grad_depth_pixel, 1);
				if constexpr (RENDER_GEOMETRY) {
					grad_normal_pixel.x = warp.shfl_up(grad_normal_pixel.x, 1);
					grad_normal_pixel.y = warp.shfl_up(grad_normal_pixel.y, 1);
					grad_normal_pixel.z = warp.shfl_up(grad_normal_pixel.z, 1);
				}
			grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
		}

		const int idx = i - static_cast<int>(lane_idx);
		const uint2 pixel_coords = {
			start_pixel_coords.x + static_cast<uint32_t>(idx % BLOCK_X),
			start_pixel_coords.y + static_cast<uint32_t>(idx / BLOCK_X)
		};
		const bool valid_pixel = pixel_coords.x < static_cast<uint32_t>(W)
			&& pixel_coords.y < static_cast<uint32_t>(H);

		if (valid_primitive && valid_pixel && lane_idx == 0 && idx >= 0 && idx < BLOCK_SIZE) {
			const int shmem_idx = i % 32;
			last_contributor = collected_last_contributor[shmem_idx];
			const float4 color_after_t = collected_color_pixel_after_transmittance[shmem_idx];
			color_pixel_after = make_float3(color_after_t);
			transmittance = color_after_t.w;
			final_trans = collected_final_trans[shmem_idx];
			depth_pixel_after = collected_depth_pixel_after[shmem_idx];
				if constexpr (RENDER_GEOMETRY) {
					const float4 geometry_after = collected_geometry_pixel_after[shmem_idx];
					depth_pixel_after = geometry_after.x;
				normal_pixel_after = make_float3(
					geometry_after.y, geometry_after.z, geometry_after.w);
			}
			const float4 grad_info = collected_grad_info_pixel[shmem_idx];
			grad_color_pixel = make_float3(grad_info);
			grad_alpha_common = grad_info.w;
			grad_depth_pixel = collected_grad_depth_pixel[shmem_idx];
				if constexpr (RENDER_GEOMETRY) {
					const float4 grad_geometry = collected_grad_geometry_pixel[shmem_idx];
				grad_alpha_common += grad_geometry.x * final_trans;
				grad_normal_pixel = make_float3(
					grad_geometry.y, grad_geometry.z, grad_geometry.w);
			}
		}

		const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= BLOCK_SIZE
			|| tile_primitive_idx >= static_cast<int>(last_contributor);
		if (skip) {
			continue;
		}

		const float2 pixel = make_float2(
			static_cast<float>(pixel_coords.x),
			static_cast<float>(pixel_coords.y));
		const float2 delta = mean2d - pixel;
		const float power = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y)
			- conic.y * delta.x * delta.y;
		if (power > 0.0f) {
			continue;
		}
		const float gaussian = expf(power);
		const float alpha = opacity * gaussian;
		if (alpha < MONO_ALPHA_THRESHOLD) {
			continue;
		}

		const float blending_weight = transmittance * alpha;

		const float3 dL_dcolor = blending_weight * grad_color_pixel;
		dL_dcolor_accum += dL_dcolor;

			float surfel_depth = depth;
			if constexpr (RENDER_GEOMETRY) {
				const float3 ray = make_float3(
				(static_cast<float>(pixel_coords.x) + 0.5f - center_x) / focal_x,
				(static_cast<float>(pixel_coords.y) + 0.5f - center_y) / focal_y,
				1.0f);
			surfel_depth = gs::surfel::rayPlaneDepth(plane, ray);
			dL_dplane_accum += gs::surfel::rayPlaneDepthBackward(
				plane, ray, blending_weight * grad_depth_pixel);
			dL_dplane_accum.x += blending_weight * grad_normal_pixel.x;
			dL_dplane_accum.y += blending_weight * grad_normal_pixel.y;
			dL_dplane_accum.z += blending_weight * grad_normal_pixel.z;
		}
		else {
			dL_ddepth_accum += blending_weight * grad_depth_pixel;
		}

		color_pixel_after -= blending_weight * color;
		depth_pixel_after -= blending_weight * surfel_depth;
		if constexpr (RENDER_GEOMETRY) {
			normal_pixel_after.x -= blending_weight * plane.x;
			normal_pixel_after.y -= blending_weight * plane.y;
			normal_pixel_after.z -= blending_weight * plane.z;
		}

		const float one_minus_alpha = 1.0f - alpha;
		const float one_minus_alpha_rcp = 1.0f / fmaxf(one_minus_alpha, MONO_ONE_MINUS_ALPHA_EPS);
		if constexpr (CAPTURE_FISHER_DIAG) {
	mono_accumulate_fisher_diag_pixel(
		primitive_idx,
		P,
		means3D,
		cov3Ds,
		viewmatrix,
		focal_x,
		focal_y,
		ddelx_dx,
		ddely_dy,
		tan_fovx,
		tan_fovy,
		make_float3(1.0f, 1.0f, 1.0f),
		conic,
		opacity,
				color,
				transmittance,
				final_trans * -(background.x + background.y + background.z),
				delta,
				gaussian,
				alpha,
				blending_weight,
				one_minus_alpha_rcp,
				color_pixel_after,
				fisher_dL_dmean3D != nullptr ? fisher_dL_dmean3D_accum : nullptr,
				fisher_dL_dopacity != nullptr ? &fisher_dL_dopacity_accum : nullptr);
		}

		const float dL_dalpha_from_color = dot(
			transmittance * color - color_pixel_after * one_minus_alpha_rcp,
			grad_color_pixel);
		const float dL_dalpha_from_depth = grad_depth_pixel
			* (transmittance * surfel_depth - depth_pixel_after * one_minus_alpha_rcp);
		float dL_dalpha_from_normal = 0.0f;
		if constexpr (RENDER_GEOMETRY) {
			dL_dalpha_from_normal = dot(
				transmittance * make_float3(plane)
					- normal_pixel_after * one_minus_alpha_rcp,
				grad_normal_pixel);
		}
		const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
		const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_depth
			+ dL_dalpha_from_normal + dL_dalpha_from_alpha;
		const float dL_dgaussian = opacity * dL_dalpha;
		

		dL_dopacity_accum += gaussian * dL_dalpha;

		const float gaussian_grad_helper = -alpha * dL_dalpha;
		dL_dconic_accum += 0.5f * gaussian_grad_helper * make_float3(
			delta.x * delta.x,
			delta.x * delta.y,
			delta.y * delta.y);
		const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
			conic.x * delta.x + conic.y * delta.y,
			conic.y * delta.x + conic.z * delta.y);
		dL_dmean2d_accum.x += dL_dmean2d.x * ddelx_dx;
		dL_dmean2d_accum.y += dL_dmean2d.y * ddely_dy;

		transmittance *= one_minus_alpha;
	}

	if (valid_primitive) {
		if (dL_dmean2D != nullptr) {
			atomicAdd(&dL_dmean2D[primitive_idx].x, dL_dmean2d_accum.x);
			atomicAdd(&dL_dmean2D[primitive_idx].y, dL_dmean2d_accum.y);
		}
		if (dL_dconic2D != nullptr) {
			atomicAdd(&dL_dconic2D[primitive_idx].x, dL_dconic_accum.x);
			atomicAdd(&dL_dconic2D[primitive_idx].y, dL_dconic_accum.y);
			atomicAdd(&dL_dconic2D[primitive_idx].w, dL_dconic_accum.z);
		}
		if (dL_dopacity != nullptr) {
			atomicAdd(&dL_dopacity[primitive_idx], dL_dopacity_accum);
		}
		if (dL_dcolors != nullptr) {
			atomicAdd(&dL_dcolors[primitive_idx * C + 0], dL_dcolor_accum.x);
			atomicAdd(&dL_dcolors[primitive_idx * C + 1], dL_dcolor_accum.y);
			atomicAdd(&dL_dcolors[primitive_idx * C + 2], dL_dcolor_accum.z);
		}
		if (dL_ddepths != nullptr) {
			atomicAdd(&dL_ddepths[primitive_idx], dL_ddepth_accum);
		}
		if (dL_dplanes != nullptr) {
			atomicAdd(&dL_dplanes[primitive_idx].x, dL_dplane_accum.x);
			atomicAdd(&dL_dplanes[primitive_idx].y, dL_dplane_accum.y);
			atomicAdd(&dL_dplanes[primitive_idx].z, dL_dplane_accum.z);
			atomicAdd(&dL_dplanes[primitive_idx].w, dL_dplane_accum.w);
		}
		
		if constexpr (CAPTURE_FISHER_DIAG) {
			monoAtomicAddArray(
				fisher_dL_dmean3D != nullptr ? fisher_dL_dmean3D + 3 * primitive_idx : nullptr,
				fisher_dL_dmean3D_accum,
				3);
			monoAtomicAddScalar(
				fisher_dL_dopacity != nullptr ? fisher_dL_dopacity + primitive_idx : nullptr,
				fisher_dL_dopacity_accum);
		}
	}
}

void MONO_BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs_dc,
	const float* shs_rest,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	const float4* dL_dplanes,
	float* dL_dcov3D,
	float* dL_dsh_dc,
	float* dL_dsh_rest,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	mono_computeCov2DCUDA_bwd << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dtau);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	mono_preprocessCUDA_bwd<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs_dc,
		shs_rest,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dplanes,
		dL_dcov3D,
		dL_dsh_dc,
		dL_dsh_rest,
		dL_dscale,
		dL_drot,
		dL_dtau);
}

void MONO_BACKWARD::render(
	const int n_buckets,
	const uint2* ranges,
	const uint32_t* tile_bucket_offsets,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float4* planes,
	const float* rendered_color,
	const float* rendered_depth,
	const float* rendered_opacity,
	const float* rendered_normal,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const uint32_t* tile_max_n_processed,
	const uint32_t* bucket_tile_index,
	const float4* bucket_color_transmittance,
	const float* bucket_depth_accum,
	const float4* bucket_geometry_accum,
	const float* dL_dpixels,
	const float* dL_dpixels_depth,
	const float* dL_dpixels_opacity,
	const float* dL_dpixels_normal,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths,
	float4* dL_dplanes,
	float* fisher_dL_dmean3D,
	float* fisher_dL_dopacity,
	int P,
	int D,
	int M,
	const float3* means3D,
	const float* shs_dc,
	const float* shs_rest,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* view,
	const float* proj,
	const float focal_x,
	const float focal_y,
	const float center_x,
	const float center_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos)
{
	if (n_buckets <= 0) {
		return;
	}

	const bool capture_fisher_diag =
		fisher_dL_dmean3D != nullptr || fisher_dL_dopacity != nullptr;
#define MONO_BUCKET_BACKWARD_LAUNCH(CAPTURE, GEOMETRY, FISHER_MEAN, FISHER_OPACITY) \
	mono_renderCUDA_bucket_bwd<NUM_CHANNELS, CAPTURE, GEOMETRY><<<n_buckets, 32>>>( \
		ranges,                                                                        \
		tile_bucket_offsets,                                                           \
		point_list,                                                                    \
		W, H,                                                                          \
		bg_color,                                                                      \
		means2D,                                                                       \
		conic_opacity,                                                                 \
		colors,                                                                        \
		depths,                                                                        \
		planes,                                                                        \
		rendered_color,                                                                \
		rendered_depth,                                                                \
		rendered_opacity,                                                              \
		rendered_normal,                                                               \
		final_Ts,                                                                      \
		n_contrib,                                                                     \
		tile_max_n_processed,                                                          \
		bucket_tile_index,                                                             \
		bucket_color_transmittance,                                                    \
		bucket_depth_accum,                                                            \
		bucket_geometry_accum,                                                         \
		dL_dpixels,                                                                    \
		dL_dpixels_depth,                                                              \
		dL_dpixels_opacity,                                                            \
		dL_dpixels_normal,                                                             \
		dL_dmean2D,                                                                    \
		dL_dconic2D,                                                                   \
		dL_dopacity,                                                                   \
		dL_dcolors,                                                                    \
		dL_ddepths,                                                                    \
		dL_dplanes,                                                                    \
				FISHER_MEAN,                                                                   \
		FISHER_OPACITY,                                                                \
		P,                                                                             \
		D,                                                                             \
		M,                                                                             \
		means3D,                                                                       \
		shs_dc,                                                                        \
		shs_rest,                                                                      \
		clamped,                                                                       \
		scales,                                                                        \
		rotations,                                                                     \
		scale_modifier,                                                                \
		cov3Ds,                                                                        \
		view,                                                                          \
		proj,                                                                          \
		focal_x,                                                                       \
		focal_y,                                                                       \
		center_x,                                                                      \
		center_y,                                                                      \
		tan_fovx,                                                                      \
		tan_fovy,                                                                      \
		campos)

	if (capture_fisher_diag) {
		if (planes != nullptr) {
			MONO_BUCKET_BACKWARD_LAUNCH(
				true, true, fisher_dL_dmean3D, fisher_dL_dopacity);
		}
		else {
			MONO_BUCKET_BACKWARD_LAUNCH(
				true, false, fisher_dL_dmean3D, fisher_dL_dopacity);
		}
	}
	else if (planes != nullptr) {
		MONO_BUCKET_BACKWARD_LAUNCH(false, true, nullptr, nullptr);
	}
	else {
		MONO_BUCKET_BACKWARD_LAUNCH(false, false, nullptr, nullptr);
	}
#undef MONO_BUCKET_BACKWARD_LAUNCH
}

