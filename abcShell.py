import numpy as np

def _lens(vecs):
    """Get the lengths of a bunch of vectors
    Use the einsum instead of the dot product because its
    it's almost always faster than np.dot
    """
    return np.sqrt(np.einsum('...ij,...ij->...i', vecs, vecs))


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
    tris = faces[fvIdxs]

    # also get the face that each tri comes from
    faceIdxs = np.repeat(np.arange(tris.size), counts - 2)

    return tris, fvIdxs, faceIdxs


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
        counts = np.full((len(faces) // 3))

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


def getEdgePairIdxs(counts, faces):
    """Build a 2*N array of ordered edge pairs"""
    # We're going to build a list of how many steps to take to get
    # the next face-vertex. Most of those values will be 1 (meaning
    # one step to the right), but at the end of each face, we need
    # to step left back to the beginning

    # Start with all 1's
    idAry = np.ones(len(faces), dtype=int)
    # Get the indices of the last index of each face
    ends = np.r_[0, counts[:-1].cumsum()]
    # Get how many steps back it is to the beginning of the face
    idAry[ends] = -counts
    # apply that offset to an arange to get the index of the next face-vert

    # Now add the step value to the index value of each spot in the array
    # to get the real indices of the next value around each face
    return np.arange(len(faces), dtype=int) + idAry


def findInArray(needles, haystack):
    """ Find the indices of values in an array

    Arguments:
        needles (np.array): The values to search for
        haystack (np.array): The array to search in

    Returns:
        np.array: The index of needle in haystack, or -1 if not found
    """
    contains = np.in1d(needles, haystack)
    defaults = np.full(needles.shape, -1)
    needles = needles[contains]

    sorter = np.argsort(haystack)
    ret = sorter[np.searchsorted(haystack, needles, sorter=sorter)]

    defaults[contains] = ret
    return defaults


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
    faceVertIdxBorderPairs = findFaceVertBorderPairs(counts, faces)
    edges = faces[faceVertIdxBorderPairs]
    stopVals = np.setdiff1d(edges[:, 1], edges[:, 0], assume_unique=True)
    if stopVals.size == 0:
        return edges[:, 0], edges
    stopIdxs = findInArray(stopVals, edges[:, 1])
    # I hope this is how 3dsMax does the order of these guys
    bVerts = np.insert(edges[:, 0], stopIdxs, edges[stopIdxs, 1])
    return bVerts, edges


def buildCycleBridges(
    startBorderVerts, endBorderVerts, cycle, bridgeFirstIdx, numBridgeSegs
):
    """Build a bridge between a starting set of edges, and an ending set of edges
    We assume that the starting/ending edge sets have the same order
    We also assume that there are no edges along the border that don't form full
    cycles
    This can be used for both Verts and UVs

    Arguments:
        startBorderVerts (np.array): A Flat array of vertex indices that define
            border edges for the starting loop
        endBorderVerts (np.array): A Flat array of vertex indices that define
            border edges for the ending loop
        cycle (np.array): An array that defines the index of the vertex that's
            the clockwise neighbor of the given index
        bridgeFirstIdx (int): Where to start counting for newly created bridge indices
        numBridgeSegs (int): The number of segments that the bridge will have between
            the start and end. It has a minimum of 1

    Returns:
        np.array: The Face array that was created. This should extend the existing
            face array for whatever mesh you're working on
        np.array: The Count array that was created. This should extend the existing
            count array for whatever mesh you're working on
    """
    numBorders = len(startBorderVerts)

    oge, ogs = np.ogrid[:numBorders, :numBridgeSegs]
    bFaces = np.zeros((numBorders, numBridgeSegs, 4), dtype=int)

    # Build the raw bridge indices
    # For simplicity of code, I don't special-case the parts where
    # I'm going to connect back to the original meshes
    bFaces[:, :, 0] = oge + (numBorders * (ogs - 1))
    bFaces[:, :, 1] = cycle[oge] + (numBorders * (ogs - 1))
    bFaces[:, :, 2] = cycle[oge] + (numBorders * ogs)
    bFaces[:, :, 3] = oge + (numBorders * ogs)

    # Offset these indices
    bFaces += bridgeFirstIdx

    # Connect them back to the original meshes
    bFaces[:, 0, 0] = startBorderVerts[oge].flatten()
    bFaces[:, 0, 1] = startBorderVerts[cycle[oge]].flatten()
    bFaces[:, -1, 2] = endBorderVerts[cycle[oge]].flatten()
    bFaces[:, -1, 3] = endBorderVerts[oge].flatten()
    bFaces = bFaces.flatten()

    # Counts is just a bunch of 4's
    bCounts = np.full((numBorders * numBridgeSegs), 4)

    return bFaces, bCounts


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
    cycle = findInArray(edges[:, 1], bVerts)

    bridgeFirstIdx = numUVs * 2

    eVerts = cycle + bridgeFirstIdx + (len(bVerts) * numBridgeSegs)
    bFaces, bCounts = buildCycleBridges(
        bVerts, eVerts, cycle, bridgeFirstIdx, numBridgeSegs
    )

    iUvFaces = reverseFaces(oUvCounts, oUvFaces) + numUVs
    faces = np.concatenate((oUvFaces, iUvFaces, bFaces))
    counts = np.concatenate((oUvCounts, oUvCounts, bCounts), axis=0)

    return faces, counts


def shellUvPos(uvs, bIdxs, prevIdxs, nxtIdxs, numBridgeSegs, offset):
    """Build the shell uv positions

    Arguments:
        uvs (np.array): N*2 array of uv positions
        bIdxs (np.array): Int array of the uv indices along the border
            that are being shelled
        prevIdxs (np.array): Int array of the uv indices counter clockwise
            from the bIdxs. If there is no vertex counterclockwise, then -1
        nxtIdxs (np.array): Int array of the uv indices clockwise
            from the bIdxs. If there is no vertex clockwise, then -1
        numBridgeSegs (int): The number of segments on the bridge geo
        offset (float): How far to perpendicularly offset the edges
            in UV space

    Returns:
        np.array: N*2 array of uv positions with inner/outer shells
            and borders
    """
    # Build the inner/outer layers, and reserve a chunk of memory
    # for the bridge points
    bVertCount = numBridgeSegs * bIdxs.length()
    bUvs = np.zeros((bVertCount, 2))
    ret = np.concatenate((uvs, uvs, bUvs))  # TODO: Don't reverse the iFaces for UV's

    # Figure out the outer vert positions, and set them in the reserved space
    prevVecs = _norm(uvs[prevIdxs] - uvs[bIdxs])
    nxtVecs = _norm(uvs[nxtIdxs] - uvs[bIdxs])
    halfAngles = np.arctan2(_lens(prevVecs + nxtVecs), _lens(prevVecs - nxtVecs))
    rotVecs = halfAngles * nxtVecs  # TODO: This isn't how to do this. Fix it

    noPrevs = prevIdxs == -1
    noNxts = nxtIdxs == -1
    rotVecs[noPrevs] = nxtVecs[noPrevs] * -90  # TODO: This isn't how to do this. Fix it
    rotVecs[noNxts] = prevVecs[noNxts] * 90  # TODO: This isn't how to do this. Fix it

    scales = offset * 2 / np.cos(halfAngles)
    scales[scales > 20] = 20.0  # set a max value
    rotVecs *= scales[..., None]
    outerVerts = rotVecs + uvs[bIdxs]
    ret[-len(outerVerts) :] = outerVerts

    # The inner vert positions are easy
    innerVerts = ret[bIdxs]

    # Interpolate between them
    for segIdx in range(1, numBridgeSegs):
        perc = float(segIdx) / numBridgeSegs
        ring = innerVerts * (1.0 - perc) + outerVerts * perc
        ringIdxs = bIdxs + ((2 * len(uvs)) + (len(bIdxs) * (segIdx - 1)))
        ret[ringIdxs] = ring

    return ret


def shellTopo(faceVertIdxBorderPairs, oFaces, oCounts, vertCount, numBridgeSegs):
    """
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
    """
    bVerts, edges = findSortedBorderEdges(oCounts, oFaces, faceVertIdxBorderPairs)
    cycle = findInArray(edges[:, 1], bVerts)

    brEdges = np.full(numBridgeSegs * len(bVerts), 4, dtype=int)
    counts = np.concatenate((oCounts, oCounts, brEdges), axis=0)

    numBorders = len(bVerts)
    offset = 2 * vertCount

    oge, ogs = np.ogrid[:numBorders, :numBridgeSegs]
    bFaces = np.zeros((numBorders, numBridgeSegs, 4), dtype=int)

    # Build the bridge indices
    bFaces[:, :, 0] = offset + oge + (numBorders * (ogs - 1))
    bFaces[:, :, 1] = offset + cycle[oge] + (numBorders * (ogs - 1))
    bFaces[:, :, 2] = offset + cycle[oge] + (numBorders * ogs)
    bFaces[:, :, 3] = offset + oge + (numBorders * ogs)

    # Connect them back to the original meshes
    bFaces[:, 0, 0] = bVerts[oge].flatten()
    bFaces[:, 0, 1] = bVerts[cycle[oge]].flatten()
    bFaces[:, -1, 2] = bVerts[cycle[oge]].flatten() + vertCount
    bFaces[:, -1, 3] = bVerts[oge].flatten() + vertCount

    iFaces = reverseFaces(oCounts, oFaces) + vertCount
    faces = np.concatenate((oFaces, iFaces, bFaces))

    return counts, faces, bVerts


def shellVertPos(rawAnim, normals, bIdxs, numBridgeSegs, innerOffset, outerOffset):
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
    numFrames = len(rawAnim)

    bVertCount = (numBridgeSegs - 1) * bIdxs.length()
    bVerts = np.zeros((numFrames, bVertCount, 3))
    ret = np.concatenate(
        (
            rawAnim + normals * outerOffset,
            rawAnim - normals * innerOffset,
            bVerts,
        )
    )

    innerVerts = ret[:, bIdxs]
    outerVerts = ret[:, bIdxs + len(rawAnim)]
    for segIdx in range(1, numBridgeSegs):
        perc = float(segIdx) / numBridgeSegs
        ring = innerVerts * (1.0 - perc) + outerVerts * perc
        ringIdxs = bIdxs + ((2 * len(rawAnim)) + (len(bIdxs) * (segIdx - 1)))
        ret[:, ringIdxs] = ring

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
    padded = anim.ndims == 2
    if padded:
        anim = anim[None, ...]

    if numBridgeSegs < 1:
        raise ValueError("The minimum number of bridge segments is 1. Got {0}".format(numBridgeSegs))

    normals = buildNormals(anim, counts, faces, triangulate=True)

    faceVertIdxBorderPairs = findFaceVertBorderPairs(counts, faces)
    bIdxs, edges = findSortedBorderEdges(counts, faces, faceVertIdxBorderPairs)
    prevIdxs = []  # TODO
    nxtIdxs = []  # TODO

    outUvFaces, outUvCounts = shellUvTopo(
        faceVertIdxBorderPairs, uvFaces, counts, len(uvs), numBridgeSegs
    )
    outUvs = shellUvPos(uvs, bIdxs, prevIdxs, nxtIdxs, numBridgeSegs, uvOffset)
    outVertCounts, outVertFaces, _ = shellTopo(
        faceVertIdxBorderPairs, faces, counts, len(anim[1]), numBridgeSegs
    )
    outAnim = shellVertPos(anim, normals, bIdxs, numBridgeSegs, innerOffset, outerOffset)

    if padded:
        outAnim = outAnim[0]

    return outAnim, outUvs, outVertCounts, outVertFaces, outUvFaces



if __name__ == "__main__":
    import json

    # Why json? Because fuck you flake8 for now allowing local ignores
    jsInput = json.loads("""{
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

    jsCheck = json.loads("""{
        "anim": [
            [-0.57071, 0.57071, 0.500], [0.000, 0.600, 0.500], [0.57071, 0.57071, 0.500],
            [-0.57071, 0.57071, 0.000], [0.000, 0.600, 0.000], [0.57071, 0.57071, 0.000],
            [-0.55773, 0.55773, -0.55773], [0.000, 0.57071, -0.57071], [0.55773, 0.55773, -0.55773],
            [-0.57071, 0.000, -0.57071], [0.000, 0.000, -0.600], [0.57071, 0.000, -0.57071],
            [-0.55773, -0.55773, -0.55773], [0.000, -0.57071, -0.57071], [0.55773, -0.55773, -0.55773],
            [-0.57071, -0.57071, 0.000], [0.000, -0.600, 0.000], [0.57071, -0.57071, 0.000],
            [-0.57071, -0.57071, 0.500], [0.000, -0.600, 0.500], [0.57071, -0.57071, 0.500],
            [-0.600, 0.000, 0.500], [0.600, 0.000, 0.500], [0.600, 0.000, 0.000],
            [-0.600, 0.000, 0.000], [-0.500, 0.500, 0.500], [0.000, 0.500, 0.500],
            [0.500, 0.500, 0.500], [-0.500, 0.500, 0.000], [0.000, 0.500, 0.000],
            [0.500, 0.500, 0.000], [-0.500, 0.500, -0.500], [0.000, 0.500, -0.500],
            [0.500, 0.500, -0.500], [-0.500, 0.000, -0.500], [0.000, 0.000, -0.500],
            [0.500, 0.000, -0.500], [-0.500, -0.500, -0.500], [0.000, -0.500, -0.500],
            [0.500, -0.500, -0.500], [-0.500, -0.500, 0.000], [0.000, -0.500, 0.000],
            [0.500, -0.500, 0.000], [-0.500, -0.500, 0.500], [0.000, -0.500, 0.500],
            [0.500, -0.500, 0.500], [-0.500, 0.000, 0.500], [0.500, 0.000, 0.500],
            [0.500, 0.000, 0.000], [-0.500, 0.000, 0.000], [0.000, 0.56666, 0.500],
            [0.54714, 0.54714, 0.500], [-0.54714, -0.54714, 0.500], [0.000, -0.56666, 0.500],
            [0.54714, -0.54714, 0.500], [0.56666, 0.000, 0.500], [-0.56666, 0.000, 0.500],
            [-0.54714, 0.54714, 0.500], [0.000, 0.53333, 0.500], [0.52357, 0.52357, 0.500],
            [-0.52357, -0.52357, 0.500], [0.000, -0.53333, 0.500], [0.52357, -0.52357, 0.500],
            [0.53333, 0.000, 0.500], [-0.53333, 0.000, 0.500], [-0.52357, 0.52357, 0.500]
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
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
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
    outerOffset = 0.1
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



