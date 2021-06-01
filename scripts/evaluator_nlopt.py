from optimizer.nlopt_optimizer import NloptOptimizer


def evaluate_nlopt():

    edge_image = _construct_edge_image(
        image_shape, camera_translator, target_genome, fitting_geometry, noise_strategy
    )

    nlopt_optimizer = NloptOptimizer(
        fitness_strategy, edge_image, start_camera.dna, geometry, nlopt_algorithm, headless
    )
    nlopt_result = nlopt_optimizer.optimize()
    logger.info("NLOPT {} took {}", nlopt_result.result_code, nlopt_result.optimize_duration)
