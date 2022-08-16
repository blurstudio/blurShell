from maya import cmds
import maya.OpenMaya as om
from dcc.maya.mayaToNumpy import mayaToNumpy, numpyToMaya
import numpy as np


# MAYA ONLY
def getDagPath(objName):
    """Get the MDagPath of an object, given its name"""
    sel = om.MSelectionList()
    sel.add(objName)
    dagPath = om.MDagPath()
    sel.getDagPath(0, dagPath)
    return dagPath


def getMeshDesc(objName):
    """Get a full numpy description of a mesh"""
    dp = getDagPath(objName)
    meshFn = om.MFnMesh(dp)

    verts = om.MPointArray()
    meshFn.getPoints(verts)
    verts = mayaToNumpy(verts)
    verts = verts[:, :3]

    counts, faces = om.MIntArray(), om.MIntArray()
    meshFn.getVertices(counts, faces)
    counts = mayaToNumpy(counts)
    faces = mayaToNumpy(faces)

    uvNames = []
    meshFn.getUVSetNames(uvNames)

    us = om.MFloatArray()
    vs = om.MFloatArray()
    meshFn.getUVs(us, vs, uvNames[0])
    us = mayaToNumpy(us)
    vs = mayaToNumpy(vs)
    uvs = np.stack((us, vs)).T

    uvCounts = om.MIntArray()
    uvFaces = om.MIntArray()
    meshFn.getAssignedUVs(uvCounts, uvFaces, uvNames[0])
    uvFaces = mayaToNumpy(uvFaces)

    return counts, verts, faces, uvs, uvFaces


def createRawObject(name, counts, verts, faces, uvs=None, uvFaces=None):
    """Build a mesh given the raw numpy data"""
    dup = cmds.polyPlane(name=name, constructionHistory=False)[0]
    dagPath = getDagPath(dup)
    fnMesh = om.MFnMesh(dagPath)

    counts = numpyToMaya(counts, om.MIntArray)
    faces = numpyToMaya(faces, om.MIntArray)

    nv = np.zeros((len(verts), 4))
    nv[:, :3] = verts
    verts = numpyToMaya(nv, om.MFloatPointArray)

    fnMesh.createInPlace(verts.length(), counts.length(), verts, counts, faces)
    fnMesh.updateSurface()

    if uvs is not None:
        us = numpyToMaya(uvs[:, 0], om.MFloatArray)
        vs = numpyToMaya(uvs[:, 1], om.MFloatArray)
        uvFaces = numpyToMaya(uvFaces, om.MIntArray)
        fnMesh.setUVs(us, vs)
        fnMesh.assignUVs(counts, uvFaces)

    cmds.sets(dup, edit=True, forceElement='initialShadingGroup')
    return dup


# NUMPY ONLY
def _dot(a, b):
    """A much faster dot-product
    Using this explicitly does better than `np.dot` and `(a*b).sum(axis=-1)`
    with any array size, but *ESPECIALLY* when a and b are more than 2-dim
    """
    return np.einsum('...ij,...ij->...i', a, b)


def _lens(vecs):
    """Get the lengths of a bunch of vectors"""
    return np.sqrt(_dot(vecs, vecs))


def _triangulate(counts, faces):
    """Given the counts and faces of a mesh, return
    a faces array of the triangulated mesh

    Arguments:
        counts (np.array): The number of vertices per face
        faces (np.array): The indices of vertices per face, flattened

    Returns:
        np.array: An N*3 array of vertex indices that will build a fully
            triangulated mesh
    """
    maxC = counts.max()
    remap = np.zeros(maxC + (maxC - 3) * 2, dtype=int)
    rng = np.arange(len(remap) / 3, dtype=int)
    remap[1::3] = rng + 1
    remap[2::3] = rng + 2
    recounts = counts + (counts - 3) * 2
    rccs = np.r_[0, recounts.cumsum()]
    smear = np.arange(rccs[-1]) - rccs[:-1].repeat(recounts)
    repc = np.r_[0, counts.cumsum()][:-1]
    out = remap[smear] + repc.repeat(recounts)
    return faces[out.reshape((-1, 3))]


def _norm(vecs):
    """Normalize an array of vectors"""
    return vecs / _lens(vecs)[..., None]


def buildNormals(anim, counts, faces, triangulate=True):
    """Build the per-vertex normals for a mesh
    Arguments:
        anim (np.array): A (frames, vertIdx, component) array of vertex positions
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert
        triangulate (bool): Whether to triangulate the mesh before building
            the normals.

    Returns:
        np.array: A (frames, vertIdx, component) array of per-vertex normals
    """

    if triangulate:
        tris = _triangulate(counts, faces)
        counts = np.ones(len(tris), dtype=int) * 3
        faces = tris.flatten()

    # Get all adjacent vertex triples on the mesh
    nextStep = np.arange(len(faces)) + 1

    ranges = np.r_[0, counts.cumsum()]
    ends = ranges[1:] - 1
    nextStep[ends] = ranges[:-1]

    trips = np.empty((len(faces), 3), dtype=int)
    step = faces[nextStep]
    trips[:, 0] = faces
    trips[:, 1] = step
    trips[:, 2] = step[nextStep]

    # Get each edge as animated vectors
    anim = anim.swapaxes(0, 1)
    v1 = anim[trips[:, 0]] - anim[trips[:, 1]]
    v2 = anim[trips[:, 2]] - anim[trips[:, 1]]

    # Possibly get the triangle areas for weighting
    # Technically I should divide by 2, but since the ratio is what matters
    # I don't actually need to do that

    # I may or may not use this in the final product
    # It depends on how 3dsMax builds the shell
    # triAreas = _lens(np.cross(v1, v2))

    # TODO: Handle zero vectors

    # Normalize the vectors
    v1 = _norm(v1)
    v2 = _norm(v2)

    # expand normalizing the face-vert Normals
    # because we can use the crossProd length to determine
    # the angle between the edges
    fvn = np.cross(v2, v1)
    fvnLens = _lens(fvn)
    fvn = fvn / fvnLens[..., None]
    angles = np.arcsin(fvnLens)

    # Weight by the angles
    weighted = fvn * angles[..., None]

    # Now use a ufunc to sum things based on the center vert
    out = np.zeros(anim.shape)
    np.add.at(out, trips[:, 1], weighted)
    out = _norm(out)
    out[np.isnan(out)] = 0.0
    out = out.swapaxes(0, 1)
    return out


def findBorderEdges(counts, faces):
    """Find all pairs of points that are border edges

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: An Nx2 numpy array of edge pairs in proper
            face order.

    """
    nextStep = np.arange(len(faces)) + 1

    ranges = np.r_[0, counts.cumsum()]
    ends = ranges[1:] - 1
    nextStep[ends] = ranges[:-1]

    edges = np.empty((len(faces), 2), dtype=int)
    edges[:, 0] = faces
    edges[:, 1] = faces[nextStep]

    # keep track of which indices will be reversed by the sort
    # flipped = edges[:, 0] > edges[:, 1]

    sedges = edges.copy()
    sedges.sort(axis=1)

    # Here's some more numpy magic
    # Re-interpret the array of edges with a new datatype
    # This datatype will be `void`, and will have memory for *the entire row*
    # This means I can use np.unique to find the unique rows in the array
    # And then get the border edge indices from that
    rowType = np.dtype((np.void, sedges.dtype.itemsize * sedges.shape[1]))
    crazyArray = np.ascontiguousarray(sedges).view(rowType)
    _, inv, ct = np.unique(crazyArray, return_inverse=True, return_counts=True)
    ret = edges[ct[inv] == 1]
    return ret[np.argsort(ret[:, 0])]


def sortBorderEdgesByTriangle(bEdges, tris):
    """The 3dsMax shell modifier orders its border edges in FaceIdx order
    Then it uses the counter-clockwise-most vertex of each edge to bridge
    the inner and outer panels.

    So, I've gotta sort my border edges in the same way as Max does

    Also, max works in *triangles* ... I may be able to ignore that, but for
    now, I'm saying it's gotta be triangulated

    Arguments:
        bEdges(np.array): An Nx2 array of vertex indices defining border edges
            These edges must be in the correct order so they match the tris
            they belong to
        tris (np.array): An Nx3 array of vertex indices defining the faces
            of a mesh

    Returns:
        np.array: The face-index sorted border edge pair array
    """
    # build a numerical type that is "pairs of integers"
    pairType = np.dtype((np.void, bEdges.dtype.itemsize * 2))

    # Convert the borders to this "Pairs" type
    bCols = np.ascontiguousarray(bEdges).view(pairType).flatten()

    edgeCols = np.zeros(tris.shape, dtype=pairType)
    for i in range(3):
        # For each set of adjacent verts around all triangles
        # convert the pair to this "Pairs" type so I can find
        # all border pairs and their indices
        tc = np.ascontiguousarray(tris[:, (i, (i + 1) % 3)])
        edgeCols[:, i] = tc.view(pairType).flatten()

    edgeCols = edgeCols.flatten()
    # Now find all the border edges
    sorter = np.argsort(edgeCols)
    idxsOf = sorter[np.searchsorted(edgeCols, bCols, sorter=sorter)]
    return bEdges[np.argsort(idxsOf)]


def reverseFaces(counts, faces):
    """Reverse the winding of the faces

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert
    """
    # Build idAry to be "the number of steps left/right to get the next index"
    # So if everything was -1, that would mean "grab the previous item in the list"
    # And that works for almost everything, except when we want the next face chunk
    # Then we've gotta step from the beginning of one face, to the end of the next,
    # That means we step the sum of the two adjacent faces (minus 1)
    # Then we make sure to start at the final index of the first face
    # Then just apply that rule to jumble the values of "faces" for the output

    # Start with all -1's
    idAry = np.ones(len(faces), dtype=int) * -1
    # Get the indices of the last index of each face
    ends = np.r_[0, counts[:-1].cumsum()]
    # Get the 2-face-step sizes
    idAry[ends[1:]] = counts[1:] + counts[:-1] - 1
    # Start at the last index of the first face
    idAry[ends[0]] = counts[0] - 1
    # Apply the rules for the output
    return faces[idAry.cumsum()]


def shell(anim, faces, counts):
    tris =   _triangulate(counts, faces)
    tCounts = np.ones(len(tris), dtype=int) * 3
    tFaces = tris.flatten()

    # in 3dsMax, the Inner shell gets the higher vertIdxs
    # Then the bridge between gets the highest
    # And the bridge loop indices increase from O to I

    innerOffset = 0.1
    outerOffset = 0.1
    numBridgeLoops = 3

    norms = buildNormals(anim, tCounts, tFaces, triangulate=False)

    oAnim = anim + (outerOffset * norms)
    iAnim = anim - (innerOffset * norms)

    oCounts = counts
    iCounts = counts

    oFaces = faces
    iFaces = faces.copy() + anim.shape[1]
    iFaces = reverseFaces(iCounts, iFaces)

    counts = np.concatenate((oCounts, iCounts))
    faces = np.concatenate((oFaces, iFaces))
    anim = np.concatenate((oAnim, iAnim), axis=-2)


    obEdges = findBorderEdges(oCounts, oFaces)
    obEdges = sortBorderEdgesByTriangle(obEdges, tris)
    ibEdges = np.flip(obEdges + iAnim.shape[1], axis=1)

    obVerts = obEdges[:, 1]
    ibVerts = obVerts + iAnim.shape[1]


    sorter = np.argsort(obEdges[:, 0])
    backCycle = sorter[np.searchsorted(obEdges[:, 0], obEdges[:, 1], sorter=sorter)]
    cycle = np.empty(backCycle.shape, dtype=int)
    cycle[backCycle] = np.arange(len(backCycle))


    bvCount = len(obVerts)
    bvStart = anim.shape[1]
    curEdges = obEdges[:, 1]
    newFaces, newVerts = [], []
    for i in range(numBridgeLoops):
        v1 = curEdges
        v2 = np.arange(bvCount, dtype=int) + bvStart + (i * bvCount)
        v3 = v2[cycle]
        v4 = v1[cycle]

        newFaces.append(np.stack((v1, v2, v3, v4), axis=1).flatten())
        curEdges = v2

        perc = (i + 1.0) / (numBridgeLoops + 1.0)

        newVerts.append(anim[:, obVerts] * (1 - perc) + anim[:, ibVerts] * perc)

    v1 = curEdges
    v2 = ibEdges[:, 0]
    v3 = v2[cycle]
    v4 = v1[cycle]
    newFaces.append(np.stack((v1, v2, v3, v4), axis=1).flatten())
    newCounts = np.ones((numBridgeLoops + 1) * bvCount, dtype=int) * 4

    anim = np.concatenate([anim] + newVerts, axis=-2)
    faces = np.concatenate([faces] + newFaces)
    counts = np.concatenate((counts, newCounts))

    return anim, faces, counts


if __name__ == "__main__":
    counts, verts, faces, uvs, uvFaces = getMeshDesc('quadPanel')
    anim, faces, counts = shell(verts[None, ...], faces, counts)
    raw = createRawObject("test", counts, anim[0], faces)


