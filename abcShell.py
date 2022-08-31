import numpy as np

def _dot(A, B):
    """Get the lengths of a bunch of vectors
    Use the einsum instead of the dot product because its
    it's almost always faster than np.dot
    """
    return np.einsum('...ij,...ij->...i', A, B)


def _lens(vecs):
    """Get the lengths of a bunch of vectors
    Use the einsum instead of the dot product because its
    it's almost always faster than np.dot
    """
    return np.sqrt(_dot(vecs, vecs))


def _norm(vecs):
    """Normalize an array of vectors"""
    return vecs / _lens(vecs)[..., None]


def triangulateFaces(counts, faces):
    """Given the counts and faces of a mesh, return
    a faces array of the triangulated mesh

    Arguments:
        counts (np.array): The number of vertices per face
        faces (np.array): The indices of vertices per face, flattened

    Returns:
        np.array: An N*3 array of vertex indices that will build a fully
            triangulated mesh
        np.array: An N*3 array of indices into the faces array saying which
            face-vert became this triangle
        np.array: An array of face indices that say which polygon each
            triangle was built from
    """
    # Get the biggest polygon size
    maxC = counts.max()

    # Build a single polygon of that size
    # So like, if the polygon was size 5
    # Build the triangle fan: [0,1,2,  0,2,3,  0,3,4]
    remap = np.zeros(maxC + (maxC - 3) * 2, dtype=int)
    rng = np.arange(len(remap) / 3, dtype=int)
    remap[1::3] = rng + 1
    remap[2::3] = rng + 2

    # build an array of the number of vertices that will be
    # in each fan
    recounts = counts + (counts - 3) * 2

    # Build the indices into the original faces array
    # based on the remap and recounts variables

    # get the pairwise start/end of each fan range
    rccs = np.r_[0, recounts.cumsum()]
    # offset each fan by the cumulative number of vertices
    smear = np.arange(rccs[-1]) - rccs[:-1].repeat(recounts)
    # Get the pairwise start/end of each polygon range
    repc = np.r_[0, counts.cumsum()][:-1]
    # Add the repeated polygon offsets to the fan offsets
    # to get my final indexing into the faces array
    fvIdxs = remap[smear] + repc.repeat(recounts)
    fvIdxs = fvIdxs.reshape((-1, 3))
    # Build the final triangle output
    tris = faces[fvIdxs].flatten()

    # also get the face that each tri comes from
    # faceIdxs = np.repeat(np.arange(counts.size), counts - 2)

    return tris


def buildNormals(anim, counts, faces, triangulate):
    """Build the per-vertex normals for a mesh

    Arguments:
        anim (np.array): A (frames, vertIdx, component) array of vertex positions
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert
        triangulate (bool): Whether to triangulate the mesh first

    Returns:
        np.array: A (frames, vertIdx, component) array of per-vertex normals
    """

    if triangulate:
        faces = triangulateFaces(counts, faces)
        counts = np.full((len(faces) // 3), 3)

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

    # 3dsMax doesn't take the triangle areas into account when building the
    # normals, but I could if I wanted to
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


def reverseFaces(counts, faces):
    """Reverse the winding of the given faces

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: The new reversed flattened indices of each vert
    """
    # Build idAry to be "the number of steps left/right to get the next index"
    # So if everything was -1, that would mean "grab the previous item in the list"
    # And that works for almost everything, except when we jump around
    # The fist jump is from the first item in the face to the last item in the face
    # (which is a jump of one less than the faceCount ... ie. cm1)
    # Then once we get to the final index of the current face (ie, the second index)
    # we have to jump to the next face which is *also* one less than the current face
    # count. So we just start at index 0, and apply those jumps

    # Start with all -1's
    idAry = np.full(len(faces), -1, dtype=int)
    # build the counts minus 1 array
    cm1 = counts - 1
    # Get the indices of the starts of each face
    starts = np.r_[0, counts.cumsum()]
    # Get the indices of the ends of each face
    ends = starts - 1
    # put the jumps to the start
    idAry[starts[:-1]] = cm1
    # put the jumps to the end
    idAry[ends[1:]] = cm1
    # evaluate the jumps
    idxs = np.r_[0, idAry.cumsum()][:-1]
    # and apply them
    return faces[idxs]


def getEdgePairIdxs(counts, faces):
    """Build a 2*N array of ordered edge pairs"""
    # We're going to build a list of how many steps to take to get
    # the next face-vertex. Most of those values will be 1 (meaning
    # one step to the right), but at the end of each face, we need
    # to step left back to the beginning

    # Start with all 1's
    idAry = np.ones(len(faces), dtype=int)
    # Get the indices of the last index of each face
    # ends = np.r_[0, counts[:-1].cumsum()]
    ends = counts.cumsum() - 1
    # Get how many steps back it is to the beginning of the face
    idAry[ends] = -(counts - 1)
    # apply that offset to an arange to get the index of the next face-vert

    # Now add the step value to the index value of each spot in the array
    # to get the real indices of the next value around each face
    return np.arange(len(faces), dtype=int) + idAry


def findInArray(needles, haystack):
    """ Find the indices of values in an array

    Arguments:
        needles (np.array): The values to search for. May have any shape
        haystack (np.array): The array to search in. Must be 1d

    Returns:
        np.array: The index of needle in haystack, or -1 if not found
    """
    nshape = needles.shape
    needles = needles.flatten()

    contains = np.in1d(needles, haystack)
    defaults = np.full(needles.shape, -1)
    needles = needles[contains]

    sorter = np.argsort(haystack)
    ret = sorter[np.searchsorted(haystack, needles, sorter=sorter)]

    defaults[contains] = ret
    return defaults.reshape(nshape)
# ---------------------------------------------------------------------


def findFaceVertBorderPairs(counts, faces):
    """Find all border edges and their face indices
    But make sure they're in the order that 3dsMax would provide them

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: An N*2 array of properly ordered vertex pairs
        np.array: An N*2 array of indices into the faces array to
            say which face-verts became these indices
        np.array: An array of length N of the face index that
            each vertex pair originated from
    """
    nextIdxs = getEdgePairIdxs(counts, faces)
    edgeDict = {}
    for eIdx, nIdx in enumerate(nextIdxs):
        a = faces[eIdx]
        b = faces[nIdx]
        key = (a, b) if a < b else (b, a)
        if key in edgeDict:
            del edgeDict[key]
        else:
            edgeDict[key] = eIdx

    startIdxs = np.array(sorted(edgeDict.values()))
    endIdxs = nextIdxs[startIdxs]

    faceVertIdxBorderPairs = np.stack((startIdxs, endIdxs), axis=1)
    return faceVertIdxBorderPairs


def findSortedBorderEdges(counts, faces, faceVertIdxBorderPairs):
    """Return the border verts and edge pairs in 3dsMax order

    Arguments:
        counts (np.array): Flat array of verts per face
        faces (np.array): Flat array of face indices
        faceVertIdxBorderPairs (np.array): N*2 array of indices
            into the faces array that build border pairs. This
            is the return from findFaceVertBorderPairs, and will
            work for both verts and UVs

    Returns:
        np.array: Flat array of border vertices in order
        np.array: N*2 array of edge pairs
    """
    edges = faces[faceVertIdxBorderPairs]
    stopVals = np.setdiff1d(edges[:, 1], edges[:, 0], assume_unique=True)
    if stopVals.size == 0:
        return edges[:, 0], edges
    stopIdxs = findInArray(stopVals, edges[:, 1])
    # I hope this is how 3dsMax does the order of these guys
    bVerts = np.insert(edges[:, 0], stopIdxs, edges[stopIdxs, 1])
    return bVerts, edges


def buildBridgesByEdge(bVerts, edges, bridgeFirstIdx, numSegs):
    """ Build the bridge faces

    Arguments:
        bVerts (np.array): The starting border vert indices
        eVerts (np.array or None): The ending border vert indices.
            It's possible to pass None to this if the end verts will
            be floating. This will be for UVs
        edges (np.array): The paired border edges in an N*2 array
        bridgeFirstIdx (int): The first vert index of the bridge
        numSegs (int): The number of segments connecting the inner
            and outer shells

    Returns:
        np.array: The flat array of new faces
        np.array: The flat array of new counts per face
    """
    # Get the indices into the bVerts
    edgeInsertIdxs = findInArray(edges, bVerts)
    edgeInsertIdxs = np.flip(edgeInsertIdxs, axis=1)

    # Build an array that I can think of like a vertIdx grid
    # Then I can populate the reference Idxs into it
    # -1 is an unfilled value
    grid = np.full((numSegs + 1, len(bVerts)), -1)
    grid[0] = bVerts

    ptr = bridgeFirstIdx
    chunkShape = (numSegs + 1, 2)
    # loop through the edges, and anywhere there's a -1
    # fill it with new vertices
    for inss in edgeInsertIdxs:
        chunk = grid[:, inss].flatten()
        toFill = chunk == -1
        fillCount = np.count_nonzero(toFill)
        chunk[toFill] = np.arange(fillCount) + ptr
        grid[:, inss] = chunk.reshape(chunkShape)
        ptr += fillCount

    # now that I've got the grid, use the edgeInsertIdxs
    # to grab the quads and return
    faces = np.stack(
        (
            grid[:, edgeInsertIdxs[:, 1]][1:],
            grid[:, edgeInsertIdxs[:, 0]][1:],
            grid[:, edgeInsertIdxs[:, 0]][:-1],
            grid[:, edgeInsertIdxs[:, 1]][:-1],
        ),
        axis=-1,
    )

    faces = faces.swapaxes(0, 1).flatten()
    counts = np.full((len(edges) * numSegs), 4)
    return faces, counts, grid


def buildBridgesByVert(bVerts, eVerts, edges, bridgeFirstIdx, numSegs):
    """ Build the bridge faces

    Arguments:
        bVerts (np.array): The starting border vert indices
        eVerts (np.array or None): The ending border vert indices.
            It's possible to pass None to this if the end verts will
            be floating. This will be for UVs
        edges (np.array): The paired border edges in an N*2 array
        bridgeFirstIdx (int): The first vert index of the bridge
        numSegs (int): The number of segments connecting the inner
            and outer shells

    Returns:
        np.array: The flat array of new faces
        np.array: The flat array of new counts per face
    """


    grid = np.full((numSegs + 1, len(bVerts)), -1)
    grid[0] = bVerts
    grid[-1] = eVerts
    for i in range(1, numSegs):
        grid[i] = np.arange(len(bVerts)) + bridgeFirstIdx + ((i - 1) * len(bVerts))

    # Get the indices into the bVerts
    edgeInsertIdxs = findInArray(edges, bVerts)
    edgeInsertIdxs = np.flip(edgeInsertIdxs, axis=1)

    # now that I've got the grid, use the edgeInsertIdxs
    # to grab the quads and return
    faces = np.stack(
        (
            grid[:, edgeInsertIdxs[:, 1]][1:],
            grid[:, edgeInsertIdxs[:, 0]][1:],
            grid[:, edgeInsertIdxs[:, 0]][:-1],
            grid[:, edgeInsertIdxs[:, 1]][:-1],
        ),
        axis=-1,
    )

    faces = faces.swapaxes(0, 1).flatten()
    counts = np.full((len(edges) * numSegs), 4)
    return faces, counts, grid


def shellUvTopo(faceVertIdxBorderPairs, oUvFaces, oUvCounts, numUVs, numBridgeSegs):
    """
    Arguments:
        faceVertIdxBorderPairs (np.array): N*2 array of indices
            into the faces array that build border pairs. This
            is the return from findFaceVertBorderPairs, and will
            work for both verts and UVs
        oUvFaces (np.ndarray): The numpy array of faces for UV indices
        oUvCounts (np.ndarray): The numpy array of counts for UV indices
        numUVs (int): The number of UVs
        numBridgeSegs (int): The number of segments that the bridge will have between
            the start and end. It has a minimum of 1
    """
    bVerts, edges = findSortedBorderEdges(oUvCounts, oUvFaces, faceVertIdxBorderPairs)

    bridgeFirstIdx = numUVs * 2
    bFaces, bCounts, grid = buildBridgesByEdge(bVerts, edges, bridgeFirstIdx, numBridgeSegs)

    iUvFaces = reverseFaces(oUvCounts, oUvFaces) + numUVs
    faces = np.concatenate((oUvFaces, iUvFaces, bFaces))
    counts = np.concatenate((oUvCounts, oUvCounts, bCounts), axis=0)

    return faces, counts, grid, edges, bVerts


def shellTopo(faceVertIdxBorderPairs, oFaces, oCounts, vertCount, numBridgeSegs):
    """ Build the vertex topology arrays for the shell

    Arguments:
        faceVertIdxBorderPairs (np.array): N*2 array of indices
            into the faces array that build border pairs. This
            is the return from findFaceVertBorderPairs, and will
            work for both verts and UVs
        oFaces (np.ndarray): The numpy array of faces for vert indices
        oCounts (np.ndarray): The numpy array of counts for vert indices
        vertCount (int): The number of verts
        numBridgeSegs (int): The number of segments that the bridge will have between
            the start and end. It has a minimum of 1

    Returns:
        np.array: The number of verts per face
        np.array: The indices of each face
        np.array: The ordered vertices on the borders
    """
    bVerts, edges = findSortedBorderEdges(oCounts, oFaces, faceVertIdxBorderPairs)

    eVerts = bVerts + vertCount
    bridgeFirstIdx = 2 * vertCount

    bFaces, bCounts, grid = buildBridgesByVert(bVerts, eVerts, edges, bridgeFirstIdx, numBridgeSegs)

    iFaces = reverseFaces(oCounts, oFaces) + vertCount
    faces = np.concatenate((oFaces, iFaces, bFaces))
    counts = np.concatenate((oCounts, oCounts, bCounts), axis=0)

    return counts, faces, grid, bVerts


def _getOffsettedUvs(uvs, grid, edges, offset):
    # The given edges are at the outside of the current shells
    # Flip 'em so they're at the inside of the new bridge
    edges = np.flip(edges, axis=1)
    bVerts = grid[0]

    nxtFind = findInArray(bVerts, edges[:, 0])
    nxtIdxs = edges[nxtFind, 1]
    nxtIdxs[nxtFind == -1] = -1
    noNxts = nxtIdxs == -1

    prevFind = findInArray(bVerts, edges[:, 1])
    prevIdxs = edges[prevFind, 0]
    prevIdxs[prevFind == -1] = -1
    noPrevs = prevIdxs == -1


    midPts = ~noPrevs & ~noNxts


    # Figure out the outer vert positions, and set them in the reserved space
    prevVecs = uvs[prevIdxs] - uvs[bVerts]
    nxtVecs = uvs[nxtIdxs] - uvs[bVerts]

    # The only way I get zero length vecs is if I'm using the same idx
    # twice (via -1), so I can just ignore that part, and use the noPrevs/noNxts

    prevVecs = _norm(prevVecs)
    nxtVecs = _norm(nxtVecs)


    mPrevs = prevVecs[midPts]
    mNxts = nxtVecs[midPts]



    halfAngles = np.arctan2(_lens(mPrevs - mNxts), _lens(mPrevs + mNxts))






    cosAngles = np.cos(halfAngles)
    sinAngles = np.sin(halfAngles)
    # Just for viewing:
    cosAngles[np.abs(cosAngles) < 1.0e-6] = 0
    sinAngles[np.abs(sinAngles) < 1.0e-6] = 0

    rotMat = np.full((len(nxtVecs), 2, 2), np.nan)
    rotMat[midPts, 0, 0] = cosAngles
    rotMat[midPts, 0, 1] = -sinAngles
    rotMat[midPts, 1, 1] = cosAngles
    rotMat[midPts, 1, 0] = sinAngles

    r90 = np.array([[[0, -1], [1, 0]]])
    rotMat[noPrevs] = r90
    rotMat[noNxts] = -r90


    toRot = nxtVecs
    toRot[noNxts] = prevVecs[noNxts]
    rotVecs = np.einsum('ij, ijk -> ik', toRot, rotMat)


    scales = np.full(len(rotVecs), offset)
    scales[midPts] = offset * 2 / (1 - np.cos(halfAngles * 2))
    scales[scales > 20] = 20.0  # set a max value

    rotVecs *= scales[..., None]
    outerVerts = rotVecs + uvs[bVerts]

    return outerVerts


def shellUvGridPos(uvs, grid, edges, offset):
    """Build the shell uv positions

    Arguments:
        uvs (np.array): N*2 array of uv positions
        grid (np.array): The grid of border and bridge uv idxs
        edges (np.array): N*2 array of paired uv indices around the border
        offset (float): The amount to offset the border edges in UV space
    Returns:
        np.array: N*2 array of uv positions with inner/outer shells
            and borders
    """
    bVerts = grid[0]
    eVerts = grid[-1]
    numBridgeSegs = grid.shape[0] - 1

    # Build the inner/outer layers, and reserve a chunk of memory
    # for the bridge points
    bUvs = np.zeros((numBridgeSegs * len(bVerts), 2), dtype=float)
    ret = np.concatenate((uvs, uvs, bUvs))

    innerVerts = uvs[bVerts]
    outerVerts = _getOffsettedUvs(uvs, grid, edges, offset)
    ret[eVerts] = outerVerts
    for segIdx in range(1, numBridgeSegs):
        perc = float(segIdx) / (len(grid) - 1)
        ret[grid[segIdx]] = innerVerts * (1.0 - perc) + outerVerts * perc

    return ret


def shellVertGridPos(rawAnim, normals, grid, innerOffset, outerOffset):
    """ Build the vertex positions for the shell

    Arguments:
        rawAnim (np.array): F*N*3 array of point positions
        normals (np.array): F*N*3 array of normal vectors
        bIdxs (np.array): The indices of the border vertices
        numBridgeSegs (int): The number of segments for the bridge
        innerOffset (float): The distance to move the inside of the shell
        outerOffset (float): The distance to move the outside of the shell

    Returns:
        np.array: F*N*3 array of new point positions
    """
    bVerts = grid[0]
    eVerts = grid[-1]
    numBridgeSegs = grid.shape[0] - 1

    # Build the inner/outer layers, and reserve a chunk of memory
    # for the bridge points
    bVertCount = (numBridgeSegs - 1) * len(bVerts)
    bVals = np.zeros((len(rawAnim), bVertCount, 3), dtype=float)
    ret = np.concatenate(
        (
            rawAnim - normals * outerOffset,
            rawAnim + normals * innerOffset,
            bVals,
        ),
        axis=1,
    )

    innerVerts = ret[:, bVerts]
    outerVerts = ret[:, eVerts]
    for segIdx in range(1, numBridgeSegs):
        perc = float(segIdx) / numBridgeSegs
        ret[:, grid[segIdx]] = innerVerts * (1.0 - perc) + outerVerts * perc

    return ret


def shell(
    anim,
    uvs,
    counts,
    faces,
    uvFaces,
    innerOffset,
    outerOffset,
    uvOffset,
    numBridgeSegs,
):
    """ Extrude flat planes of geometry, give them thickness,
    and add a number of loops to thickened edges

    Arguments:
        anim (np.array): A N*3 or F*N*3 array of point positions
        uvs (np.array): A N*2 array of uv positions
        counts (np.array): A flat array of vert counts per face
        faces (np.array): A flat array of vert indices, grouped by count
        uvFaces (np.array): A flat array of uv indices, grouped by count
        innerOffset (float): The amount to move the inner side
        outerOffset (float): The amount to move the outer side
        uvOffset (float): The width of the bridge in UV space
        numBridgeSegs (int): The number of segments to split the bridge
            into. Its minimum value is 1

    Returns:
        np.array: The vertex positions with the same number of dimensions
            as the `anim` input
        np.array: The UV positions
        np.array: The new counts array
        np.array: The new faces array
        np.array: The new uvFaces array
    """
    padded = anim.ndim == 2
    if padded:
        anim = anim[None, ...]

    if numBridgeSegs < 1:
        raise ValueError("The minimum number of bridge segments is 1. Got {0}".format(numBridgeSegs))

    normals = buildNormals(anim, counts, faces, triangulate=True)

    faceVertIdxBorderPairs = findFaceVertBorderPairs(counts, faces)

    # Do the uvs
    outUvFaces, outUvCounts, uvGrid, uvEdges, bUvIdxs = shellUvTopo(
        faceVertIdxBorderPairs, uvFaces, counts, len(uvs), numBridgeSegs
    )
    outUvs = shellUvGridPos(uvs, uvGrid, uvEdges, uvOffset)


    # Do the faces
    outVertCounts, outVertFaces, vertGrid, bIdxs = shellTopo(
        faceVertIdxBorderPairs, faces, counts, anim.shape[1], numBridgeSegs
    )
    outAnim = shellVertGridPos(anim, normals, vertGrid, innerOffset, outerOffset)

    if padded:
        outAnim = outAnim[0]

    return outAnim, outUvs, outVertCounts, outVertFaces, outUvFaces




def _test():
    import json

    # Why json? Because fuck you flake8 for now allowing local ignores

    # 2x2x2 box with missing bottom face
    jsInput1 = json.loads("""{
        "anim" : [
            [-0.5, -0.5,  0.5], [ 0.0, -0.5,  0.5], [ 0.5, -0.5,  0.5],
            [-0.5,  0.0,  0.5], [ 0.0,  0.0,  0.5], [ 0.5,  0.0,  0.5],
            [-0.5,  0.5,  0.5], [ 0.0,  0.5,  0.5], [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.0], [ 0.0,  0.5,  0.0], [ 0.5,  0.5,  0.0],
            [-0.5,  0.5, -0.5], [ 0.0,  0.5, -0.5], [ 0.5,  0.5, -0.5],
            [-0.5,  0.0, -0.5], [ 0.0,  0.0, -0.5], [ 0.5,  0.0, -0.5],
            [-0.5, -0.5, -0.5], [ 0.0, -0.5, -0.5], [ 0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.0], [ 0.5, -0.5,  0.0], [ 0.5,  0.0,  0.0],
            [-0.5,  0.0,  0.0]
        ],

        "uvs" : [
            [0.275, 0.1  ], [0.4,   0.1  ], [0.725, 0.1  ], [0.275, 0.225],
            [0.6,   0.225], [0.725, 0.225], [0.275, 0.35 ], [0.6,   0.35 ],
            [0.725, 0.35 ], [0.275, 0.475], [0.6,   0.475], [0.725, 0.475],
            [0.275, 0.6  ], [0.6,   0.6  ], [0.725, 0.6  ], [0.275, 0.725],
            [0.6,   0.725], [0.725, 0.725], [0.275, 0.85 ], [0.6,   0.85 ],
            [0.725, 0.85 ], [0.975, 0.1  ], [0.85,  0.1  ], [0.975, 0.225],
            [0.85,  0.225], [0.975, 0.35 ], [0.85,  0.35 ], [0.025, 0.1  ],
            [0.15,  0.1  ], [0.025, 0.225], [0.15,  0.225], [0.025, 0.35 ],
            [0.15,  0.35 ], [0.4,   0.725], [0.4,   0.85 ], [0.4,   0.6  ],
            [0.4,   0.475], [0.4,   0.35 ], [0.4,   0.225], [0.6,   0.1  ]
        ],

        "counts" : [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],

        "faces" : [
            3,4,1,0,      4,5,2,1,      6,7,4,3,      7,8,5,4,
            9,10,7,6,     10,11,8,7,    12,13,10,9,   13,14,11,10,
            15,16,13,12,  16,17,14,13,  18,19,16,15,  19,20,17,16,
            23,17,20,22,  5,23,22,2,    11,14,17,23,  8,11,23,5,
            15,24,21,18,  24,3,0,21,    12,9,24,15,   9,6,3,24
        ],

        "uvFaces" : [
            3,38,1,0,     4,5,2,39,     6,37,38,3,    7,8,5,4,
            9,36,37,6,    10,11,8,7,    12,35,36,9,   13,14,11,10,
            15,33,35,12,  16,17,14,13,  18,34,33,15,  19,20,17,16,
            24,23,21,22,  5,24,22,2,    26,25,23,24,  8,26,24,5,
            29,30,28,27,  30,3,0,28,    31,32,30,29,  32,6,3,30
        ]
    }""")

    jsCheck1 = json.loads("""{
        "anim": [
            [-0.57071,  0.57071,  0.500],   [ 0.000,    0.600,    0.500],   [0.57071,   0.57071,  0.500],
            [-0.57071,  0.57071,  0.000],   [ 0.000,    0.600,    0.000],   [0.57071,   0.57071,  0.000],
            [-0.55773,  0.55773, -0.55773], [ 0.000,    0.57071, -0.57071], [0.55773,   0.55773, -0.55773],
            [-0.57071,  0.000,   -0.57071], [ 0.000,    0.000,   -0.600],   [0.57071,   0.000,   -0.57071],
            [-0.55773, -0.55773, -0.55773], [ 0.000,   -0.57071, -0.57071], [0.55773,  -0.55773, -0.55773],
            [-0.57071, -0.57071,  0.000],   [ 0.000,   -0.600,    0.000],   [0.57071,  -0.57071,  0.000],
            [-0.57071, -0.57071,  0.500],   [ 0.000,   -0.600,    0.500],   [0.57071,  -0.57071,  0.500],
            [-0.600,    0.000,    0.500],   [ 0.600,    0.000,    0.500],   [0.600,     0.000,    0.000],
            [-0.600,    0.000,    0.000],   [-0.500,    0.500,    0.500],   [0.000,     0.500,    0.500],
            [ 0.500,    0.500,    0.500],   [-0.500,    0.500,    0.000],   [0.000,     0.500,    0.000],
            [ 0.500,    0.500,    0.000],   [-0.500,    0.500,   -0.500],   [0.000,     0.500,   -0.500],
            [ 0.500,    0.500,   -0.500],   [-0.500,    0.000,   -0.500],   [0.000,     0.000,   -0.500],
            [ 0.500,    0.000,   -0.500],   [-0.500,   -0.500,   -0.500],   [0.000,    -0.500,   -0.500],
            [ 0.500,   -0.500,   -0.500],   [-0.500,   -0.500,    0.000],   [0.000,    -0.500,    0.000],
            [ 0.500,   -0.500,    0.000],   [-0.500,   -0.500,    0.500],   [0.000,    -0.500,    0.500],
            [ 0.500,   -0.500,    0.500],   [-0.500,    0.000,    0.500],   [0.500,     0.000,    0.500],
            [ 0.500,    0.000,    0.000],   [-0.500,    0.000,    0.000],   [0.000,     0.56666,  0.500],
            [ 0.54714,  0.54714,  0.500],   [-0.54714, -0.54714,  0.500],   [0.000,    -0.56666,  0.500],
            [ 0.54714, -0.54714,  0.500],   [ 0.56666,  0.000,    0.500],   [-0.56666,  0.000,    0.500],
            [-0.54714,  0.54714,  0.500],   [ 0.000,    0.53333,  0.500],   [0.52357,   0.52357,  0.500],
            [-0.52357, -0.52357,  0.500],   [ 0.000,   -0.53333,  0.500],   [0.52357,  -0.52357,  0.500],
            [ 0.53333,  0.000,    0.500],   [-0.53333,  0.000,    0.500],   [-0.52357,  0.52357,  0.500]
        ],
        "uvs": [
            [0.275, 0.100], [0.400, 0.100], [0.725, 0.100], [0.275, 0.225],
            [0.600, 0.225], [0.725, 0.225], [0.275, 0.350], [0.600, 0.350],
            [0.725, 0.350], [0.275, 0.475], [0.600, 0.475], [0.725, 0.475],
            [0.275, 0.600], [0.600, 0.600], [0.725, 0.600], [0.275, 0.725],
            [0.600, 0.725], [0.725, 0.725], [0.275, 0.850], [0.600, 0.850],
            [0.725, 0.850], [0.975, 0.100], [0.850, 0.100], [0.975, 0.225],
            [0.850, 0.225], [0.975, 0.350], [0.850, 0.350], [0.025, 0.100],
            [0.150, 0.100], [0.025, 0.225], [0.150, 0.225], [0.025, 0.350],
            [0.150, 0.350], [0.400, 0.725], [0.400, 0.850], [0.400, 0.600],
            [0.400, 0.475], [0.400, 0.350], [0.400, 0.225], [0.600, 0.100],
            [0.275, 0.100], [0.400, 0.100], [0.725, 0.100], [0.275, 0.225],
            [0.600, 0.225], [0.725, 0.225], [0.275, 0.350], [0.600, 0.350],
            [0.725, 0.350], [0.275, 0.475], [0.600, 0.475], [0.725, 0.475],
            [0.275, 0.600], [0.600, 0.600], [0.725, 0.600], [0.275, 0.725],
            [0.600, 0.725], [0.725, 0.725], [0.275, 0.850], [0.600, 0.850],
            [0.725, 0.850], [0.975, 0.100], [0.850, 0.100], [0.975, 0.225],
            [0.850, 0.225], [0.975, 0.350], [0.850, 0.350], [0.025, 0.100],
            [0.150, 0.100], [0.025, 0.225], [0.150, 0.225], [0.025, 0.350],
            [0.150, 0.350], [0.400, 0.725], [0.400, 0.850], [0.400, 0.600],
            [0.400, 0.475], [0.400, 0.350], [0.400, 0.225], [0.600, 0.100],
            [0.275, 0.08333], [0.400, 0.08333], [0.275, 0.06666], [0.400, 0.06666],
            [0.275, 0.050], [0.400, 0.050], [0.600, 0.08333], [0.725, 0.08333],
            [0.600, 0.06666], [0.725, 0.06666], [0.600, 0.050], [0.725, 0.050],
            [0.400, 0.86666], [0.275, 0.86666], [0.400, 0.88333], [0.275, 0.88333],
            [0.400, 0.900], [0.275, 0.900], [0.725, 0.86666], [0.600, 0.86666],
            [0.725, 0.88333], [0.600, 0.88333], [0.725, 0.900], [0.600, 0.900],
            [0.850, 0.08333], [0.975, 0.08333], [0.850, 0.06666], [0.975, 0.06666],
            [0.850, 0.050], [0.975, 0.050], [0.025, 0.08333], [0.150, 0.08333],
            [0.025, 0.06666], [0.150, 0.06666], [0.025, 0.050], [0.150, 0.050]
        ],
        "counts": [
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
        ],
        "faces": [
            1, 0, 3, 4,        2, 1, 4, 5,        4, 3, 6, 7,        5, 4, 7, 8,
            7, 6, 9, 10,       8, 7, 10, 11,      10, 9, 12, 13,     11, 10, 13, 14,
            13, 12, 15, 16,    14, 13, 16, 17,    16, 15, 18, 19,    17, 16, 19, 20,
            20, 22, 23, 17,    22, 2, 5, 23,      17, 23, 11, 14,    23, 5, 8, 11,
            21, 18, 15, 24,    0, 21, 24, 3,      24, 15, 12, 9,     3, 24, 9, 6,
            26, 29, 28, 25,    27, 30, 29, 26,    29, 32, 31, 28,    30, 33, 32, 29,
            32, 35, 34, 31,    33, 36, 35, 32,    35, 38, 37, 34,    36, 39, 38, 35,
            38, 41, 40, 37,    39, 42, 41, 38,    41, 44, 43, 40,    42, 45, 44, 41,
            45, 42, 48, 47,    47, 48, 30, 27,    42, 39, 36, 48,    48, 36, 33, 30,
            46, 49, 40, 43,    25, 28, 49, 46,    49, 34, 37, 40,    28, 31, 34, 49,
            50, 57, 0, 1,      58, 65, 57, 50,    26, 25, 65, 58,    51, 50, 1, 2,
            59, 58, 50, 51,    27, 26, 58, 59,    52, 53, 19, 18,    60, 61, 53, 52,
            43, 44, 61, 60,    53, 54, 20, 19,    61, 62, 54, 53,    44, 45, 62, 61,
            54, 55, 22, 20,    62, 63, 55, 54,    45, 47, 63, 62,    55, 51, 2, 22,
            63, 59, 51, 55,    47, 27, 59, 63,    56, 52, 18, 21,    64, 60, 52, 56,
            46, 43, 60, 64,    57, 56, 21, 0,     65, 64, 56, 57,    25, 46, 64, 65
        ],
        "uvFaces": [
            1, 0, 3, 38,          2, 39, 4, 5,          38, 3, 6, 37,         5, 4, 7, 8,
            37, 6, 9, 36,         8, 7, 10, 11,         36, 9, 12, 35,        11, 10, 13, 14,
            35, 12, 15, 33,       14, 13, 16, 17,       33, 15, 18, 34,       17, 16, 19, 20,
            21, 22, 24, 23,       22, 2, 5, 24,         23, 24, 26, 25,       24, 5, 8, 26,
            28, 27, 29, 30,       0, 28, 30, 3,         30, 29, 31, 32,       3, 30, 32, 6,
            41, 78, 43, 40,       42, 45, 44, 79,       78, 77, 46, 43,       45, 48, 47, 44,
            77, 76, 49, 46,       48, 51, 50, 47,       76, 75, 52, 49,       51, 54, 53, 50,
            75, 73, 55, 52,       54, 57, 56, 53,       73, 74, 58, 55,       57, 60, 59, 56,
            61, 63, 64, 62,       62, 64, 45, 42,       63, 65, 66, 64,       64, 66, 48, 45,
            68, 70, 69, 67,       40, 43, 70, 68,       70, 72, 71, 69,       43, 46, 72, 70,
            81, 80, 0, 1,         83, 82, 80, 81,       85, 84, 82, 83,       87, 86, 39, 2,
            89, 88, 86, 87,       91, 90, 88, 89,       93, 92, 34, 18,       95, 94, 92, 93,
            97, 96, 94, 95,       99, 98, 20, 19,       101, 100, 98, 99,     103, 102, 100, 101,
            105, 104, 22, 21,     107, 106, 104, 105,   109, 108, 106, 107,   104, 87, 2, 22,
            106, 89, 87, 104,     108, 91, 89, 106,     111, 110, 27, 28,     113, 112, 110, 111,
            115, 114, 112, 113,   80, 111, 28, 0,       82, 113, 111, 80,     84, 115, 113, 82
        ]
    }""")


    # 1x1x2 box with missing bottom face
    jsInput2 = json.loads("""{
        "anim": [
            [-5.0,  0.0,  5.0], [5.0,  0.0,  5.0], [-5.0,  0.0,  0.0], [5.0,  0.0,  0.0],
            [-5.0,  0.0, -5.0], [5.0,  0.0, -5.0], [-5.0, 10.0,  5.0], [5.0, 10.0,  5.0],
            [-5.0, 10.0,  0.0], [5.0, 10.0,  0.0], [-5.0, 10.0, -5.0], [5.0, 10.0, -5.0]
        ],
        "uvs": [
            [0.375, 0.325], [0.625, 0.325], [0.375, 0.450], [0.625, 0.450],
            [0.375, 0.675], [0.625, 0.675], [0.375, 0.075], [0.625, 0.075],
            [0.875, 0.325], [0.875, 0.450], [0.875, 0.675], [0.625, 0.925],
            [0.375, 0.925], [0.125, 0.675], [0.125, 0.550], [0.125, 0.325],
            [0.375, 0.550], [0.625, 0.550], [0.875, 0.550], [0.125, 0.450]
        ],
        "counts": [4, 4, 4, 4, 4, 4, 4, 4],
        "faces": [
            8,9,7,6,   10,11,9,8,  6,7,1,0,   7,9,3,1,
            9,11,5,3,  11,10,4,5,  10,8,2,4,  8,6,0,2
        ],
        "uvFaces": [
            2,3,1,0,     4,5,17,16,  0,1,7,6,     1,3,9,8,
            17,5,10,18,  5,4,12,11,  4,16,14,13,  2,0,15,19
        ]
    }""")

    jsCheck2 = json.loads("""{
        "anim": [
            [-5.7071, 0.0, 5.7071], [5.7071, 0.0, 5.7071], [-6.0, 0.0, 0.0], [6.0, 0.0, 0.0],
            [-5.7071, 0.0, -5.7071], [5.7071, 0.0, -5.7071], [-5.57735, 10.57735, 5.57735],
            [5.57735, 10.57735, 5.57735], [-5.7071, 10.7071, 0.0], [5.7071, 10.7071, 0.0],
            [-5.57735, 10.57735, -5.57735], [5.57735, 10.57735, -5.57735],
            [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 0.0], [5.0, 0.0, 0.0],
            [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [-5.0, 10.0, 5.0], [5.0, 10.0, 5.0],
            [-5.0, 10.0, 0.0], [5.0, 10.0, 0.0], [-5.0, 10.0, -5.0], [5.0, 10.0, -5.0],
            [5.47140, 0.0, 5.47140], [5.6666, 0.0, 0.0], [5.47140, 0.0, -5.47140],
            [-5.47140, 0.0, -5.47140], [-5.6666, 0.0, 0.0], [-5.47140, 0.0, 5.47140],
            [5.23570, 0.0, 5.23570], [5.3333, 0.0, 0.0], [5.23570, 0.0, -5.23570],
            [-5.23570, 0.0, -5.23570], [-5.3333, 0.0, 0.0], [-5.23570, 0.0, 5.23570]
        ],
        "uvs": [
            [0.375, 0.325], [0.625, 0.325], [0.375, 0.450], [0.625, 0.450],
            [0.375, 0.675], [0.625, 0.675], [0.375, 0.075], [0.625, 0.075],
            [0.875, 0.325], [0.875, 0.450], [0.875, 0.675], [0.625, 0.925],
            [0.375, 0.925], [0.125, 0.675], [0.125, 0.550], [0.125, 0.325],
            [0.375, 0.550], [0.625, 0.550], [0.875, 0.550], [0.125, 0.450],
            [0.375, 0.325], [0.625, 0.325], [0.375, 0.450], [0.625, 0.450],
            [0.375, 0.675], [0.625, 0.675], [0.375, 0.075], [0.625, 0.075],
            [0.875, 0.325], [0.875, 0.450], [0.875, 0.675], [0.625, 0.925],
            [0.375, 0.925], [0.125, 0.675], [0.125, 0.550], [0.125, 0.325],
            [0.375, 0.550], [0.625, 0.550], [0.875, 0.550], [0.125, 0.450],
            [0.375, 0.0583333], [0.625, 0.0583333], [0.375, 0.0416666], [0.625, 0.0416666],
            [0.375, 0.025], [0.625, 0.025], [0.8916666, 0.325], [0.8916666, 0.450],
            [0.9083333, 0.325], [0.9083333, 0.450], [0.925, 0.325], [0.925, 0.450],
            [0.8916666, 0.550], [0.8916666, 0.675], [0.9083333, 0.550], [0.9083333, 0.675],
            [0.925, 0.550], [0.925, 0.675], [0.625, 0.9416666], [0.375, 0.9416666],
            [0.625, 0.9583333], [0.375, 0.9583333], [0.625, 0.975], [0.375, 0.975],
            [0.1083333, 0.675], [0.1083333, 0.550], [0.0916666, 0.675], [0.0916666, 0.550],
            [0.075, 0.675], [0.075, 0.550], [0.1083333, 0.450], [0.1083333, 0.325],
            [0.0916666, 0.450], [0.0916666, 0.325], [0.075, 0.450], [0.075, 0.325]
        ],
        "counts": [
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
        ],
        "faces": [
            8,9,7,6,      10,11,9,8,    6,7,1,0,      7,9,3,1,
            9,11,5,3,     11,10,4,5,    10,8,2,4,     8,6,0,2,
            20,18,19,21,  22,20,21,23,  18,12,13,19,  19,13,15,21,
            21,15,17,23,  23,17,16,22,  22,16,14,20,  20,14,12,18,
            24,29,0,1,    30,35,29,24,  13,12,35,30,  25,24,1,3,
            31,30,24,25,  15,13,30,31,  26,25,3,5,    32,31,25,26,
            17,15,31,32,  27,26,5,4,    33,32,26,27,  16,17,32,33,
            28,27,4,2,    34,33,27,28,  14,16,33,34,  29,28,2,0,
            35,34,28,29,  12,14,34,35
        ],
        "uvFaces": [
            2,3,1,0,      4,5,17,16,    0,1,7,6,      1,3,9,8,
            17,5,10,18,   5,4,12,11,    4,16,14,13,   2,0,15,19,
            22,20,21,23,  24,36,37,25,  20,26,27,21,  21,28,29,23,
            37,38,30,25,  25,31,32,24,  24,33,34,36,  22,39,35,20,
            41,40,6,7,    43,42,40,41,  45,44,42,43,  47,46,8,9,
            49,48,46,47,  51,50,48,49,  53,52,18,10,  55,54,52,53,
            57,56,54,55,  59,58,11,12,  61,60,58,59,  63,62,60,61,
            65,64,13,14,  67,66,64,65,  69,68,66,67,  71,70,19,15,
            73,72,70,71,  75,74,72,73
        ]
    }""")


    # 1x1x2 box with missing bottom face not split along median
    jsInput3 = json.loads("""{
        "anim": [
            [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 0.0], [5.0, 0.0, 0.0],
            [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [-5.0, 10.0, 5.0], [5.0, 10.0, 5.0],
            [-5.0, 10.0, 0.0], [5.0, 10.0, 0.0], [-5.0, 10.0, -5.0], [5.0, 10.0, -5.0]
        ],
        "uvs": [
            [0.375, 0.375], [0.625, 0.375], [0.375, 0.5], [0.625, 0.5],
            [0.375, 0.625], [0.625, 0.625], [0.375, 0.125], [0.625, 0.125],
            [0.875, 0.375], [0.875, 0.5], [0.875, 0.625], [0.625, 0.875],
            [0.375, 0.875], [0.125, 0.625], [0.125, 0.5], [0.125, 0.375]
        ],
        "counts": [4, 4, 4, 4, 4, 4, 4, 4],
        "faces": [
            8,9,7,6,  10,11,9,8, 6,7,1,0,  7,9,3,1,
            9,11,5,3, 11,10,4,5, 10,8,2,4, 8,6,0,2
        ],
        "uvFaces": [
            2,3,1,0,  4,5,3,2,   0,1,7,6,   1,3,9,8,
            3,5,10,9, 5,4,12,11, 4,2,14,13, 2,0,15,14
        ]
    }""")

    jsCheck3 = json.loads("""{
        "anim": [
            [-5.7071, 0.0, 5.7071], [5.7071, 0.0, 5.7071], [-6.0, 0.0, 0.0], [6.0, 0.0, 0.0],
            [-5.7071, 0.0, -5.7071], [5.7071, 0.0, -5.7071], [-5.57735, 10.57735, 5.57735],
            [5.57735, 10.57735, 5.57735], [-5.7071, 10.7071, 0.0], [5.7071, 10.7071, 0.0],
            [-5.57735, 10.57735, -5.57735], [5.57735, 10.57735, -5.57735],
            [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 0.0], [5.0, 0.0, 0.0],
            [-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [-5.0, 10.0, 5.0], [5.0, 10.0, 5.0],
            [-5.0, 10.0, 0.0], [5.0, 10.0, 0.0], [-5.0, 10.0, -5.0], [5.0, 10.0, -5.0],
            [5.4714, 0.0, 5.4714], [5.6666, 0.0, 0.0], [5.4714, 0.0, -5.4714], [-5.4714, 0.0, -5.4714],
            [-5.6666, 0.0, 0.0], [-5.4714, 0.0, 5.4714], [5.2357, 0.0, 5.2357], [5.3333, 0.0, 0.0],
            [5.2357, 0.0, -5.2357], [-5.2357, 0.0, -5.2357], [-5.3333, 0.0, 0.0], [-5.2357, 0.0, 5.2357]
        ],
        "uvs": [
            [0.375, 0.375], [0.625, 0.375], [0.375, 0.5], [0.625, 0.5],
            [0.375, 0.625], [0.625, 0.625], [0.375, 0.125], [0.625, 0.125],
            [0.875, 0.375], [0.875, 0.5], [0.875, 0.625], [0.625, 0.875],
            [0.375, 0.875], [0.125, 0.625], [0.125, 0.5], [0.125, 0.375],
            [0.375, 0.375], [0.625, 0.375], [0.375, 0.5], [0.625, 0.5],
            [0.375, 0.625], [0.625, 0.625], [0.375, 0.125], [0.625, 0.125],
            [0.875, 0.375], [0.875, 0.5], [0.875, 0.625], [0.625, 0.875],
            [0.375, 0.875], [0.125, 0.625], [0.125, 0.5], [0.125, 0.375],
            [0.375, 0.1083333], [0.625, 0.1083333], [0.375, 0.0916666], [0.625, 0.0916666],
            [0.375, 0.075], [0.625, 0.075], [0.8916666, 0.375], [0.8916666, 0.5],
            [0.9083333, 0.375], [0.9083333, 0.5], [0.925, 0.375], [0.925, 0.5],
            [0.8916666, 0.625], [0.9083333, 0.625], [0.925, 0.625], [0.625, 0.8916666],
            [0.375, 0.8916666], [0.625, 0.9083333], [0.375, 0.9083333], [0.625, 0.925],
            [0.375, 0.925], [0.1083333, 0.625], [0.1083333, 0.5], [0.0916666, 0.625],
            [0.0916666, 0.5], [0.075, 0.625], [0.075, 0.5], [0.1083333, 0.375],
            [0.0916666, 0.375], [0.075, 0.375]
        ],
        "counts": [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
        "faces": [
            8,9,7,6,      10,11,9,8,    6,7,1,0,      7,9,3,1,
            9,11,5,3,     11,10,4,5,    10,8,2,4,     8,6,0,2,
            20,18,19,21,  22,20,21,23,  18,12,13,19,  19,13,15,21,
            21,15,17,23,  23,17,16,22,  22,16,14,20,  20,14,12,18,
            24,29,0,1,    30,35,29,24,  13,12,35,30,
            25,24,1,3,    31,30,24,25,  15,13,30,31,
            26,25,3,5,    32,31,25,26,  17,15,31,32,
            27,26,5,4,    33,32,26,27,  16,17,32,33,
            28,27,4,2,    34,33,27,28,  14,16,33,34,
            29,28,2,0,    35,34,28,29,  12,14,34,35
        ],
        "uvFaces": [
            2,3,1,0,      4,5,3,2,      0,1,7,6,      1,3,9,8,
            3,5,10,9,     5,4,12,11,    4,2,14,13,    2,0,15,14,
            18,16,17,19,  20,18,19,21,  16,22,23,17,  17,24,25,19,
            19,25,26,21,  21,27,28,20,  20,29,30,18,  18,30,31,16,
            33,32,6,7,    35,34,32,33,  37,36,34,35,
            39,38,8,9,    41,40,38,39,  43,42,40,41,
            44,39,9,10,   45,41,39,44,  46,43,41,45,
            48,47,11,12,  50,49,47,48,  52,51,49,50,
            54,53,13,14,  56,55,53,54,  58,57,55,56,
            59,54,14,15,  60,56,54,59,  61,58,56,60
        ]
    }""")

    jsInput = jsInput3
    jsCheck = jsCheck3




    inAnim = np.array(jsInput["anim"], dtype=float)
    inUvs = np.array(jsInput["uvs"], dtype=float)
    inCounts = np.array(jsInput["counts"], dtype=int)
    inFaces = np.array(jsInput["faces"], dtype=int)
    inUvFaces = np.array(jsInput["uvFaces"], dtype=int)

    chkAnim = np.array(jsCheck["anim"], dtype=float)
    chkUvs = np.array(jsCheck["uvs"], dtype=float)
    chkCounts = np.array(jsCheck["counts"], dtype=int)
    chkFaces = np.array(jsCheck["faces"], dtype=int)
    chkUvFaces = np.array(jsCheck["uvFaces"], dtype=int)

    innerOffset = 0.0
    outerOffset = 1.0
    uvOffset = 0.05
    numBridgeSegs = 3

    outAnim, outUvs, outCounts, outFaces, outUvFaces = shell(
        inAnim,
        inUvs,
        inCounts,
        inFaces,
        inUvFaces,
        innerOffset,
        outerOffset,
        uvOffset,
        numBridgeSegs,
    )

    closeAnim = np.isclose(outAnim, chkAnim, rtol=1e-4)
    closeUvs = np.isclose(outUvs, chkUvs, rtol=1e-4)
    closeCounts = outCounts == chkCounts
    closeFaces = outFaces == chkFaces
    closeUvFaces = outUvFaces == chkUvFaces

    closeAnimAll = np.all(closeAnim)
    closeUvsAll = np.all(closeUvs)
    closeCountsAll = np.all(closeCounts)
    closeFacesAll = np.all(closeFaces)
    closeUvFacesAll = np.all(closeUvFaces)

    assert closeAnimAll
    assert closeUvsAll
    assert closeCountsAll
    assert closeFacesAll
    assert closeUvFacesAll


if __name__ == "__main__":
    _test()
