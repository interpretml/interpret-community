# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a structure for gathering and storing the parts of an explanation asset."""


class ModelSummary():
    """A structure for gathering and storing the parts of an explanation asset."""

    def __init__(self):
        """Initialize data structures to hold summary information."""
        self.artifacts = []
        self.meta_dict = {}

    def add_from_get_model_summary(self, name, artifact_metadata_tuple):
        """Update artifacts and metadata with new information.

        :param name: The name the new data should be associated with.
        :type name: str
        :param artifact_metadata_tuple: The tuple of artifacts and metadata to add to existing.
        :type artifact_metadata_tuple: (list[dict], dict)
        """
        self.artifacts += artifact_metadata_tuple[0]
        self.meta_dict[name] = artifact_metadata_tuple[1]

    def get_artifacts(self):
        """Get the list of artifacts.

        :return: Artifact list.
        :rtype: list[list[dict]]
        """
        return self.artifacts

    def get_metadata_dictionary(self):
        """Get the combined dictionary of metadata.

        :return: Metadata dictionary.
        :rtype: dict
        """
        return self.meta_dict
