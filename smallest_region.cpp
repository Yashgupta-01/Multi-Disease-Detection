#include <bits/stdc++.h>
using namespace std;

// Function to find the minimum uncovered interval along one axis (x or y)
long long find_min_uncovered_interval(const vector<array<int, 4>>& segments, 
                                      int axis_start, int axis_end, int axis) {
    // axis: 0 for x axis, 1 for y axis
    int coord1 = (axis == 0 ? 0 : 1); // Start coordinate index
    int coord2 = (axis == 0 ? 2 : 3); // End coordinate index
    set<int> breakpoints{axis_start, axis_end}; // Set of all interesting coordinates

    // Collect all endpoints from the segments
    for (const auto& segment : segments) {
        breakpoints.insert(segment[coord1]);
        breakpoints.insert(segment[coord2]);
    }
    vector<int> sortedPoints(breakpoints.begin(), breakpoints.end());

    // Find points NOT covered by any segment
    vector<int> uncoveredPoints;
    for (int point : sortedPoints) {
        bool isCovered = false;
        for (const auto& segment : segments) {
            if (segment[coord1] < point && point < segment[coord2]) {
                isCovered = true;
                break;
            }
        }
        if (!isCovered)
            uncoveredPoints.push_back(point);
    }

    long long smallestInterval = (long long)1e18;
    // Check for smallest difference between uncovered points
    for (size_t i = 0; i + 1 < uncoveredPoints.size(); ++i) {
        int intervalLength = uncoveredPoints[i + 1] - uncoveredPoints[i];
        if (intervalLength == 1)
            return 1;
        if (intervalLength < smallestInterval)
            smallestInterval = intervalLength;
    }

    // Check for any gaps not covered fully
    for (size_t i = 0; i + 1 < sortedPoints.size(); ++i) {
        int start = sortedPoints[i], end = sortedPoints[i + 1];
        if (end - start <= 1)
            continue;
        bool fullyCovered = false;
        for (const auto& segment : segments) {
            if (segment[coord1] <= start && end <= segment[coord2]) {
                fullyCovered = true;
                break;
            }
        }
        if (!fullyCovered)
            return 1;
    }
    return smallestInterval;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int numSegments;
    cin >> numSegments;
    vector<array<int, 4>> segments(numSegments);
    for (int i = 0; i < numSegments; ++i)
        cin >> segments[i][0] >> segments[i][1] >> segments[i][2] >> segments[i][3];

    int regionXStart, regionYStart, regionXEnd, regionYEnd;
    cin >> regionXStart >> regionYStart >> regionXEnd >> regionYEnd;

    long long minWidth = find_min_uncovered_interval(segments, regionXStart, regionXEnd, 0); // 0 for x axis
    long long minHeight = find_min_uncovered_interval(segments, regionYStart, regionYEnd, 1); // 1 for y axis

    cout << (minWidth * minHeight) << "\n";
    return 0;
}
