package org.minecraftsim.harness;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import net.minecraft.server.MinecraftServer;

/**
 * Serves newline-delimited UTF-8 control JSON on a loopback-only socket.
 * Socket I/O runs on its daemon thread; commands are submitted to the server thread
 * and each response waits for completion before the next request is handled.
 */
public final class ControlSocket implements AutoCloseable {
	private static final Gson GSON = new Gson();
	private final MinecraftServer server;
	private final ControlCommands commands;
	private final ServerSocket listener;
	private volatile Socket client;
	private volatile boolean closed;

	/**
	 * Binds the configured loopback port and starts the control daemon.
	 * @param server server that executes commands
	 * @param bots shared bot manager supplied to the command dispatcher
	 * @throws IOException if the loopback listener cannot be bound
	 */
	public ControlSocket(MinecraftServer server, BotManager bots) throws IOException {
		this.server = server;
		this.commands = new ControlCommands(server, bots);
		int port = Integer.getInteger("simharness.port", 25599);
		this.listener = new ServerSocket();
		listener.setReuseAddress(true);
		listener.bind(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), port), 1);
		Thread thread = new Thread(this::serve, "simharness-control");
		thread.setDaemon(true);
		thread.start();
		SimHarness.LOGGER.info("SimHarness control listening on 127.0.0.1:{}", port);
	}

	private void serve() {
		while (!closed) {
			try (Socket socket = listener.accept()) {
				client = socket;
				socket.setTcpNoDelay(true);
				try (var reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
						var writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8))) {
					String line;
					while (!closed && (line = reader.readLine()) != null) {
						if (handle(line, writer)) {
							return;
						}
					}
				}
			} catch (IOException failure) {
				if (!closed) {
					SimHarness.LOGGER.warn("Control connection closed: {}", failure.getMessage());
				}
			} finally {
				client = null;
			}
		}
	}

	private boolean handle(String line, BufferedWriter writer) throws IOException {
		JsonElement id = JsonNull.INSTANCE;
		String command = "";
		JsonObject response = new JsonObject();
		boolean success = false;
		try {
			JsonObject request = JsonParser.parseString(line).getAsJsonObject();
			if (!request.has("id") || !request.get("id").isJsonPrimitive() || !request.getAsJsonPrimitive("id").isNumber()) {
				throw new IllegalArgumentException("id must be a number");
			}
			id = request.get("id").deepCopy();
			command = request.get("cmd").getAsString();
			JsonObject args = request.has("args") ? request.getAsJsonObject("args") : new JsonObject();
			CompletableFuture<JsonObject> result = new CompletableFuture<>();
			String submittedCommand = command;
			server.execute(() -> {
				try {
					commands.execute(submittedCommand, args).whenComplete((value, failure) -> {
						if (failure == null) {
							result.complete(value);
						} else {
							result.completeExceptionally(failure);
						}
					});
				} catch (Exception failure) {
					result.completeExceptionally(failure);
				}
			});
			response.add("result", result.join());
			success = true;
		} catch (Exception failure) {
			Throwable cause = failure;
			while (cause instanceof CompletionException && cause.getCause() != null) {
				cause = cause.getCause();
			}
			response.addProperty("error", cause.getMessage() == null ? cause.getClass().getSimpleName() : cause.getMessage());
		}
		response.add("id", id);
		response.addProperty("ok", success);
		writer.write(GSON.toJson(response));
		writer.newLine();
		writer.flush();
		if (success && command.equals("stop")) {
			server.execute(() -> server.halt(false));
			return true;
		}
		return false;
	}

	/** Closes listener/client sockets, logging rather than propagating close failures. */
	@Override
	public void close() {
		closed = true;
		try {
			listener.close();
			Socket active = client;
			if (active != null) {
				active.close();
			}
		} catch (IOException failure) {
			SimHarness.LOGGER.warn("Unable to close control socket", failure);
		}
	}
}
