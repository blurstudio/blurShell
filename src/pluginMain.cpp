#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "shell.h"
#include "version.h"

MStatus initializePlugin(MObject obj) {
	MStatus result;
	MFnPlugin plugin(obj, "Blur Studio", VERSION_STRING, "Any");
	result = plugin.registerNode(
        DEFORMER_NAME,
        shell::id,
        shell::creator,
        shell::initialize
    );
	return result;
}

MStatus uninitializePlugin(MObject obj) {
	MStatus result;
	MFnPlugin plugin(obj);
	result = plugin.deregisterNode(shell::id);
	return result;
}

