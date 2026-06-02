#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/tcp-socket-base.h"
#include "gym_tcp_env.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Topo3ParkinglotStress");

namespace
{
constexpr double STEP_TIME = 0.1;
constexpr double SIM_TIME = 160.0;
constexpr uint32_t PACKET_SIZE = 536;

const std::string RESULT_DIR = "scratch/results/parkinglot_stress_rtt/";
const std::string BOTTLENECK_RATE = "15Mbps";
const std::string QUEUE_SIZE = "150p";

std::ofstream f_cwnd_s1;
std::ofstream f_cwnd_s2;
std::ofstream f_rtt_s1;
std::ofstream f_rtt_s2;
std::ofstream f_goodput_s1;
std::ofstream f_goodput_s2;

// flowId:
// 1 = S1 FTP/background
// 2 = S2 primary video-like protected flow
// 3 = S3 burst cross traffic
// 4 = S4 additional overlapping burst cross traffic
std::array<uint64_t, 5> g_rxBytes = {0, 0, 0, 0, 0};

void OpenResultFiles(const std::string &prefix)
{
    std::filesystem::create_directories(RESULT_DIR);

    f_cwnd_s1.open(RESULT_DIR + prefix + "_cwnd_s1.txt");
    f_cwnd_s2.open(RESULT_DIR + prefix + "_cwnd_s2.txt");
    f_rtt_s1.open(RESULT_DIR + prefix + "_rtt_s1.txt");
    f_rtt_s2.open(RESULT_DIR + prefix + "_rtt_s2.txt");
    f_goodput_s1.open(RESULT_DIR + prefix + "_goodput_s1.txt");
    f_goodput_s2.open(RESULT_DIR + prefix + "_goodput_s2.txt");
}

void CloseResultFiles()
{
    f_cwnd_s1.close();
    f_cwnd_s2.close();
    f_rtt_s1.close();
    f_rtt_s2.close();
    f_goodput_s1.close();
    f_goodput_s2.close();
}

void GymStep(Ptr<GymTcpEnv> env)
{
    if (Simulator::IsFinished())
    {
        return;
    }

    env->Notify();
    Simulator::Schedule(Seconds(STEP_TIME), &GymStep, env);
}

void RxCallback(Ptr<GymTcpEnv> env, uint32_t flowId, Ptr<const Packet> packet, const Address &address)
{
    uint32_t bytes = packet->GetSize();

    // Only S1/S2 are RL-controlled and observed by the Gym environment.
    if (flowId == 1 || flowId == 2)
    {
        env->ReportRx(flowId, bytes);
    }

    if (flowId < g_rxBytes.size())
    {
        g_rxBytes[flowId] += bytes;
    }
}

void QueueDropCallback(Ptr<GymTcpEnv> env, Ptr<const Packet> packet)
{
    env->ReportQueueDrop(1);
}

void LogGoodput()
{
    double now = Simulator::Now().GetSeconds();

    double s1GoodputMbps = static_cast<double>(g_rxBytes[1]) * 8.0 / STEP_TIME / 1e6;
    double s2GoodputMbps = static_cast<double>(g_rxBytes[2]) * 8.0 / STEP_TIME / 1e6;

    if (f_goodput_s1.is_open())
    {
        f_goodput_s1 << now << " " << s1GoodputMbps << "\n";
    }

    if (f_goodput_s2.is_open())
    {
        f_goodput_s2 << now << " " << s2GoodputMbps << "\n";
    }

    g_rxBytes[1] = 0;
    g_rxBytes[2] = 0;
    g_rxBytes[3] = 0;
    g_rxBytes[4] = 0;

    if (!Simulator::IsFinished())
    {
        Simulator::Schedule(Seconds(STEP_TIME), &LogGoodput);
    }
}

void PlotCwnd1(uint32_t oldVal, uint32_t newVal)
{
    if (f_cwnd_s1.is_open())
    {
        f_cwnd_s1 << Simulator::Now().GetSeconds() << " " << newVal << "\n";
    }
}

void PlotCwnd2(uint32_t oldVal, uint32_t newVal)
{
    if (f_cwnd_s2.is_open())
    {
        f_cwnd_s2 << Simulator::Now().GetSeconds() << " " << newVal << "\n";
    }
}

void PlotRtt1(Time oldVal, Time newVal)
{
    if (f_rtt_s1.is_open() && newVal.GetMilliSeconds() > 0)
    {
        f_rtt_s1 << Simulator::Now().GetSeconds() << " " << newVal.GetMilliSeconds() << "\n";
    }
}

void PlotRtt2(Time oldVal, Time newVal)
{
    if (f_rtt_s2.is_open() && newVal.GetMilliSeconds() > 0)
    {
        f_rtt_s2 << Simulator::Now().GetSeconds() << " " << newVal.GetMilliSeconds() << "\n";
    }
}

void ConnectTcpTracers()
{
    Config::ConnectWithoutContext(
        "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
        MakeCallback(&PlotCwnd1));

    Config::ConnectWithoutContext(
        "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/RTT",
        MakeCallback(&PlotRtt1));

    Config::ConnectWithoutContext(
        "/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
        MakeCallback(&PlotCwnd2));

    Config::ConnectWithoutContext(
        "/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/RTT",
        MakeCallback(&PlotRtt2));
}

void ConnectQueueDropTracer(Ptr<GymTcpEnv> env, const NetDeviceContainer &dR2R3)
{
    // Parking-lot traffic direction:
    // S1 -> R1 -> R2 -> R3 -> D1
    // S2/S3/S4 -> R2 -> R3 -> D2/D3/D4
    // Therefore, the shared congested output queue is R2->R3, dR2R3.Get(0).
    Ptr<PointToPointNetDevice> r2Device =
        DynamicCast<PointToPointNetDevice>(dR2R3.Get(0));

    if (!r2Device || !r2Device->GetQueue())
    {
        NS_LOG_WARN("Failed to connect R2->R3 queue drop tracer.");
        return;
    }

    r2Device->GetQueue()->TraceConnectWithoutContext(
        "Drop",
        MakeCallback(&QueueDropCallback).Bind(env));
}

bool TryRegisterAgentOnce(
    Ptr<GymTcpEnv> env,
    uint32_t nodeId,
    uint32_t flowId,
    uint32_t serviceType)
{
    if (env->HasAgent(flowId))
    {
        return true;
    }

    std::ostringstream path;
    path << "/NodeList/" << nodeId
         << "/$ns3::TcpL4Protocol/SocketList/0";

    Config::MatchContainer match = Config::LookupMatches(path.str());

    if (match.GetN() == 0)
    {
        return false;
    }

    Ptr<TcpSocketBase> socket = DynamicCast<TcpSocketBase>(match.Get(0));

    if (!socket)
    {
        return false;
    }

    return env->AddTcpAgent(socket, flowId, serviceType);
}

void TryRegisterAgentWithRetry(
    Ptr<GymTcpEnv> env,
    uint32_t nodeId,
    uint32_t flowId,
    uint32_t serviceType,
    double deadline)
{
    bool success = TryRegisterAgentOnce(env, nodeId, flowId, serviceType);

    if (success)
    {
        return;
    }

    if (Simulator::Now().GetSeconds() < deadline)
    {
        Simulator::Schedule(
            Seconds(0.2),
            &TryRegisterAgentWithRetry,
            env,
            nodeId,
            flowId,
            serviceType,
            deadline);
    }
    else
    {
        NS_LOG_WARN("Failed to register RL agent before deadline. flowId=" << flowId);
    }
}

} // namespace

int main(int argc, char *argv[])
{
    std::string mode = "RL";

    CommandLine cmd;
    cmd.AddValue("mode", "Baseline or RL", mode);
    cmd.Parse(argc, argv);

    Config::SetDefault(
        "ns3::TcpL4Protocol::SocketType",
        StringValue("ns3::TcpCubic"));

    std::string prefix = (mode == "RL") ? "rl" : "baseline";
    OpenResultFiles(prefix);

    NodeContainer S;
    NodeContainer R;
    NodeContainer D;

    S.Create(4); // S1, S2, S3, S4
    R.Create(3); // R1, R2, R3
    D.Create(4); // D1, D2, D3, D4

    PointToPointHelper access;
    PointToPointHelper bottleneck;

    access.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    access.SetChannelAttribute("Delay", StringValue("2ms"));

    bottleneck.SetDeviceAttribute("DataRate", StringValue(BOTTLENECK_RATE));
    bottleneck.SetChannelAttribute("Delay", StringValue("10ms"));
    bottleneck.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue(QUEUE_SIZE));

    // Parking-lot stress topology:
    //
    // S1 ---- R1 ==== R2 ==== R3 ---- D1
    //                / |  \     \
    //              S2  S3  S4    D2/D3/D4
    //
    // S1: FTP/background, RL-controlled
    // S2: primary video-like flow, RL-controlled and protected
    // S3: burst cross traffic, 50s~100s, not RL-controlled
    // S4: additional overlapping burst cross traffic, 70s~100s, not RL-controlled
    //
    // R2->R3 is the key shared bottleneck.
    NetDeviceContainer dS1R1 = access.Install(S.Get(0), R.Get(0));
    NetDeviceContainer dR1R2 = bottleneck.Install(R.Get(0), R.Get(1));
    NetDeviceContainer dS2R2 = access.Install(S.Get(1), R.Get(1));
    NetDeviceContainer dS3R2 = access.Install(S.Get(2), R.Get(1));
    NetDeviceContainer dS4R2 = access.Install(S.Get(3), R.Get(1));
    NetDeviceContainer dR2R3 = bottleneck.Install(R.Get(1), R.Get(2));
    NetDeviceContainer dR3D1 = access.Install(R.Get(2), D.Get(0));
    NetDeviceContainer dR3D2 = access.Install(R.Get(2), D.Get(1));
    NetDeviceContainer dR3D3 = access.Install(R.Get(2), D.Get(2));
    NetDeviceContainer dR3D4 = access.Install(R.Get(2), D.Get(3));

    InternetStackHelper internet;
    internet.InstallAll();

    Ipv4AddressHelper address;

    address.SetBase("10.5.1.0", "255.255.255.0");
    address.Assign(dS1R1);

    address.SetBase("10.5.2.0", "255.255.255.0");
    address.Assign(dR1R2);

    address.SetBase("10.5.3.0", "255.255.255.0");
    address.Assign(dS2R2);

    address.SetBase("10.5.4.0", "255.255.255.0");
    address.Assign(dS3R2);

    address.SetBase("10.5.5.0", "255.255.255.0");
    address.Assign(dS4R2);

    address.SetBase("10.5.6.0", "255.255.255.0");
    address.Assign(dR2R3);

    address.SetBase("10.5.7.0", "255.255.255.0");
    Ipv4InterfaceContainer iR3D1 = address.Assign(dR3D1);

    address.SetBase("10.5.8.0", "255.255.255.0");
    Ipv4InterfaceContainer iR3D2 = address.Assign(dR3D2);

    address.SetBase("10.5.9.0", "255.255.255.0");
    Ipv4InterfaceContainer iR3D3 = address.Assign(dR3D3);

    address.SetBase("10.5.10.0", "255.255.255.0");
    Ipv4InterfaceContainer iR3D4 = address.Assign(dR3D4);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    Ptr<GymTcpEnv> env = CreateObject<GymTcpEnv>();

    PacketSinkHelper sink(
        "ns3::TcpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), 9));

    for (uint32_t i = 0; i < 4; ++i)
    {
        ApplicationContainer sinkApp = sink.Install(D.Get(i));
        sinkApp.Start(Seconds(0.0));

        sinkApp.Get(0)->TraceConnectWithoutContext(
            "Rx",
            MakeCallback(&RxCallback).Bind(env, i + 1));
    }

    // S1: FTP / background best-effort traffic
    BulkSendHelper src1(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iR3D1.GetAddress(1), 9));
    src1.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s1App = src1.Install(S.Get(0));
    s1App.Start(Seconds(0.0));
    s1App.Stop(Seconds(SIM_TIME));

    // S2: primary video-like flow, RL protected target
    BulkSendHelper src2(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iR3D2.GetAddress(1), 9));
    src2.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s2App = src2.Install(S.Get(1));
    s2App.Start(Seconds(0.0));
    s2App.Stop(Seconds(SIM_TIME));

    // S3: original burst cross traffic
    BulkSendHelper src3(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iR3D3.GetAddress(1), 9));
    src3.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s3App = src3.Install(S.Get(2));
    s3App.Start(Seconds(50.0));
    s3App.Stop(Seconds(100.0));

    // S4: additional overlapping burst cross traffic
    BulkSendHelper src4(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iR3D4.GetAddress(1), 9));
    src4.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s4App = src4.Install(S.Get(3));
    s4App.Start(Seconds(70.0));
    s4App.Stop(Seconds(100.0));

    ConnectQueueDropTracer(env, dR2R3);

    Simulator::Schedule(Seconds(STEP_TIME), &LogGoodput);
    Simulator::Schedule(Seconds(1.0), &ConnectTcpTracers);

    if (mode == "RL")
    {
        Simulator::Schedule(
            Seconds(0.2),
            &TryRegisterAgentWithRetry,
            env,
            0,
            1,
            1,
            5.0);

        Simulator::Schedule(
            Seconds(0.2),
            &TryRegisterAgentWithRetry,
            env,
            1,
            2,
            2,
            5.0);

        Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface>(5555);
        env->SetOpenGymInterface(openGym);

        Simulator::Schedule(Seconds(STEP_TIME), &GymStep, env);
    }

    NS_LOG_UNCOND("Mode: " << mode
                  << ", Topology: Parking-lot Stress"
                  << ", TCP: TcpCubic"
                  << ", Bottleneck: " << BOTTLENECK_RATE
                  << ", Queue: " << QUEUE_SIZE
                  << ", S3 burst: 50~100s"
                  << ", S4 burst: 70~100s"
                  << ", Step: " << STEP_TIME
                  << "s, SimTime: " << SIM_TIME << "s"
                  << ", ResultDir: " << RESULT_DIR);

    Simulator::Stop(Seconds(SIM_TIME));
    Simulator::Run();
    Simulator::Destroy();

    CloseResultFiles();

    return 0;
}
